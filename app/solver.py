"""Virtual-battery dispatcher.

Pure discrete optimization over integer actions.  The model (one action per
period, no simultaneous charge/discharge):

    s_{t+1} = s_t + charge_t - discharge_t
    0 <= s_t <= capacity
    0 <= charge_t    <= max_charge     (grid + PV may charge the battery)
    0 <= discharge_t <= min(max_charge, load_t - pv_t)   when positive part

Surplus PV after battery charging is curtailed (never sold).  The grid only
sells: purchase_t = max(0, load_t - pv_t - discharge_t + charge_t).

Objective, strictly lexicographically:
    1. minimum total purchase cost   sum(price * purchase)
    2. minimum total purchase amount sum(purchase)
    3. minimum charge/discharge sequence lexicographically
       (sequence ordered charge-first then discharge per period, so
       (c0, d0, c1, d1, ...); compares only on full ties of 1 and 2)

Optional purchase budget
------------------------

``purchase_budget`` is an optional non-negative integer capping the SUM of
purchase over *all* periods:  sum(purchase_t) <= purchase_budget.  It is a
first-class optimization constraint: the budgeted DP carries the cumulative
purchase amount as a state dimension (see :func:`_budgeted_dp`); we never
solve the unconstrained problem and then discard plans afterwards.  With the
budget omitted the solver behaves exactly as before.

Two structurally different kinds of infeasibility are reported separately:

* the physical constraints alone admit no plan (terminal SoC unreachable);
* every physically feasible plan needs more total purchase than the budget —
  the response then carries ``minimum_purchase``, the minimum total purchase
  of any physically feasible plan.

Neither infeasible response contains a partial "current plan".

Battery capacity is at most 50 (enforced by the API model), so exhaustive
dynamic programming over integer states is trivially fast and exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ScenarioParams:
    """Normalized scenario parameters passed to :func:`solve`."""

    load: Tuple[int, ...]
    pv: Tuple[int, ...]
    price: Tuple[int, ...]
    capacity: int
    initial_soc: int
    terminal_min_soc: int
    max_charge: int
    purchase_budget: Optional[int] = None

    @property
    def n(self) -> int:
        return len(self.load)


@dataclass(frozen=True)
class PeriodResult:
    period: int
    soc_start: int
    charge: int
    discharge: int
    soc_end: int
    net_load: int          # load - pv, may be negative
    purchase: int
    curtail: int
    cost: int
    remaining_budget: Optional[int] = None


@dataclass(frozen=True)
class Solution:
    feasible: bool
    periods: Tuple[PeriodResult, ...]
    total_cost: int
    total_purchase: int
    final_soc: Optional[int]
    reason: Optional[str] = None
    purchase_budget: Optional[int] = None
    # Minimum total purchase of any physically feasible plan; populated only
    # when infeasibility is caused solely by the purchase budget.
    minimum_purchase: Optional[int] = None


# (cost, purchase, sequence-tuple) is already the lexicographic order of the
# three objectives, so Python tuple comparison implements the tie break.
Label = Tuple[int, int, Tuple[int, ...]]

# Budgeted suffix label: cumulative purchase is a state dimension of its own,
# so the label leads with used purchase, then cost, then the action sequence.
BudgetLabel = Tuple[int, int, Tuple[int, ...]]


def _better(a: Optional[Label], b: Optional[Label]) -> Optional[Label]:
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _actions(
    p: "ScenarioParams", t: int, soc: int
) -> List[Tuple[Tuple[int, int], int, int, int]]:
    """All legal actions at state ``soc`` in period ``t``.

    Returns tuples ``(action, next_soc, purchase, cost)`` in no particular
    order (charge actions first, then discharges; idle is the c=0 entry).
    """
    cap = p.capacity
    mc = p.max_charge
    net = p.load[t] - p.pv[t]
    price = p.price[t]
    max_dis = min(mc, max(0, net), soc)
    out: List[Tuple[Tuple[int, int], int, int, int]] = []
    for c in range(0, min(mc, cap - soc) + 1):
        purchase = max(0, net + c)
        out.append(((c, 0), soc + c, purchase, price * purchase))
    for d in range(1, max_dis + 1):
        purchase = max(0, net - d)
        out.append(((0, d), soc - d, purchase, price * purchase))
    return out


def _unconstrained_dp(p: "ScenarioParams") -> List[List[Optional[Label]]]:
    """dp[t][s] = best (cost, purchase, seq) suffix label from state s at t."""
    n = p.n
    cap = p.capacity
    dp: List[List[Optional[Label]]] = [[None] * (cap + 1) for _ in range(n + 1)]
    for s in range(cap + 1):
        dp[n][s] = (0, 0, ()) if s >= p.terminal_min_soc else None

    for t in range(n - 1, -1, -1):
        for s in range(cap + 1):
            best: Optional[Label] = None
            for (c, d), ns, purchase, cost in _actions(p, t, s):
                tail = dp[t + 1][ns]
                if tail is None:
                    continue
                label = (tail[0] + cost, tail[1] + purchase, (c, d) + tail[2])
                best = _better(best, label)
            dp[t][s] = best
    return dp


def _budgeted_dp(
    p: "ScenarioParams", used_cap: int
) -> List[List[Dict[Tuple[int, int], BudgetLabel]]]:
    """Exact DP with the budget embedded as a cumulative-purchase dimension.

    dp[t][s] maps ``(used_purchase, cost)`` -> best suffix label (minimum
    sequence on ties), restricted to suffix plans whose cumulative purchase
    does not exceed ``used_cap`` (the remaining budget at state s at the
    start of t is handled by the caller: labels measure suffix usage).

    Labels dominated on BOTH purchase and cost are pruned: a suffix using no
    more purchase and costing no more always wins under the lexicographic
    objective.  ``used_cap`` is bounded by the (cheaply precomputed)
    unconstrained minimum purchase, since no plan using more purchase can
    ever be optimal under objective 2 — this is a DP state bound, not a
    post-hoc plan filter.
    """
    n = p.n
    cap = p.capacity
    dp: List[List[Dict[Tuple[int, int], BudgetLabel]]] = [
        [{} for _ in range(cap + 1)] for _ in range(n + 1)
    ]
    for s in range(cap + 1):
        if s >= p.terminal_min_soc:
            dp[n][s] = {(0, 0): (0, 0, ())}

    for t in range(n - 1, -1, -1):
        for s in range(cap + 1):
            # Collect best sequence for each (used, cost), then Pareto-prune.
            cand: Dict[Tuple[int, int], BudgetLabel] = {}
            for (c, d), ns, purchase, cost in _actions(p, t, s):
                for (tu, tk), tail in dp[t + 1][ns].items():
                    used = tu + purchase
                    if used > used_cap:
                        continue
                    key = (used, tk + cost)
                    label: BudgetLabel = (used, key[1], (c, d) + tail[2])
                    old = cand.get(key)
                    if old is None or label[2] < old[2]:
                        cand[key] = label

            # Pareto frontier: scan (used, cost) ascending; keep an entry
            # only while its (cost, sequence) improves over every suffix that
            # uses strictly/equal less purchase.  Equal (used, cost) keeps
            # the smallest sequence (already first under this ordering).
            best_key: Optional[Tuple[int, Tuple[int, ...]]] = None
            pruned: Dict[Tuple[int, int], BudgetLabel] = {}
            for key in sorted(cand):
                label = cand[key]
                marker = (key[1], label[2])
                if best_key is None or marker < best_key:
                    pruned[key] = label
                    best_key = marker
            dp[t][s] = pruned
    return dp


def _period_result(
    p: "ScenarioParams", t: int, soc: int, c: int, d: int, ns: int,
    purchase: int, cost: int, remaining_budget: Optional[int],
) -> PeriodResult:
    net = p.load[t] - p.pv[t]
    residual = net - d + c
    return PeriodResult(
        period=t,
        soc_start=soc,
        charge=c,
        discharge=d,
        soc_end=ns,
        net_load=net,
        purchase=purchase,
        curtail=max(0, -residual),
        cost=cost,
        remaining_budget=remaining_budget,
    )


def solve(p: ScenarioParams) -> Solution:
    """Exact DP with lexicographic action reconstruction.

    A dense suffix DP classifies physical feasibility (and yields the minimum
    total purchase, needed to diagnose a budget shortfall); when a budget is
    present the actual optimum under that constraint is reconstructed from a
    genuinely budget-constrained DP — never by post-filtering an
    unconstrained plan.
    """
    cap = p.capacity
    budget = p.purchase_budget

    if not (0 <= p.initial_soc <= cap):
        return Solution(False, (), 0, 0, None,
                        "initial SoC outside [0, capacity]",
                        purchase_budget=budget)

    dp = _unconstrained_dp(p)
    root = dp[0][p.initial_soc]
    if root is None:
        # Physical infeasibility: no plan reaches the terminal minimum SoC,
        # independent of any purchase budget.
        return Solution(False, (), 0, 0, None,
                        "cannot reach terminal minimum SoC within charge/discharge limits",
                        purchase_budget=budget)

    opt_cost, opt_purchase = root[0], root[1]

    if budget is None:
        return _reconstruct_unconstrained(p, dp, root)

    if opt_purchase > budget:
        # Physically feasible plans exist but all buy at least opt_purchase;
        # the budget alone makes the scenario unsolvable.  No partial plan.
        return Solution(
            False, (), 0, 0, None,
            f"purchase budget {budget} is insufficient: every physically "
            f"feasible plan requires at least {opt_purchase} total purchase",
            purchase_budget=budget,
            minimum_purchase=opt_purchase,
        )

    return _reconstruct_budgeted(p, budget, opt_purchase)


def _reconstruct_unconstrained(
    p: ScenarioParams, dp: List[List[Optional[Label]]], root: Label
) -> Solution:
    """Forward reconstruction: smallest feasible action keeping the optimum."""
    n = p.n
    soc = p.initial_soc
    remaining_cost = root[0]
    remaining_purchase = root[1]
    periods: List[PeriodResult] = []
    total_cost = 0
    total_purchase = 0

    for t in range(n):
        chosen = None
        for (c, d), ns, purchase, cost in sorted(_actions(p, t, soc), key=lambda x: x[0]):
            tail = dp[t + 1][ns]
            if tail is None:
                continue
            if (tail[0] + cost, tail[1] + purchase) == (remaining_cost, remaining_purchase):
                chosen = ((c, d), ns, purchase, cost, tail)
                break
        assert chosen is not None, "DP reconstruction failed"
        (c, d), ns, purchase, cost, tail = chosen
        periods.append(_period_result(p, t, soc, c, d, ns, purchase, cost, None))
        total_cost += cost
        total_purchase += purchase
        soc = ns
        remaining_cost, remaining_purchase = tail[0], tail[1]

    assert total_cost == root[0] and total_purchase == root[1]
    return Solution(True, tuple(periods), total_cost, total_purchase, soc)


def _reconstruct_budgeted(
    p: ScenarioParams, budget: int, minimum_purchase: int
) -> Solution:
    """Reconstruct the lexicographic optimum of the budget-constrained DP."""
    n = p.n
    # State bound: no optimal plan uses more than the unconstrained minimum.
    bdp = _budgeted_dp(p, used_cap=minimum_purchase)
    table = bdp[0][p.initial_soc]
    assert table, "budgeted DP lost a feasible plan"
    # Objective order is (cost, purchase, sequence); table keys (used, cost).
    (root_used, root_cost), root_label = min(
        table.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[1][2])
    )
    assert root_used <= budget

    soc = p.initial_soc
    used_so_far = 0
    rem_used = root_used
    rem_cost = root_cost
    rem_seq: Optional[Tuple[int, ...]] = root_label[2]
    periods: List[PeriodResult] = []
    total_cost = 0
    total_purchase = 0

    for t in range(n):
        chosen = None
        for (c, d), ns, purchase, cost in sorted(_actions(p, t, soc), key=lambda x: x[0]):
            tail_key = (rem_used - purchase, rem_cost - cost)
            tail = bdp[t + 1][ns].get(tail_key)
            if tail is None:
                continue
            seq = (c, d) + tail[2]
            if rem_seq is not None and seq != rem_seq:
                continue
            chosen = ((c, d), ns, purchase, cost, tail)
            break
        assert chosen is not None, "budgeted DP reconstruction failed"
        (c, d), ns, purchase, cost, tail = chosen

        used_so_far += purchase
        remaining_budget = budget - used_so_far
        assert remaining_budget >= 0
        periods.append(_period_result(
            p, t, soc, c, d, ns, purchase, cost, remaining_budget
        ))
        total_cost += cost
        total_purchase += purchase
        soc = ns
        rem_used, rem_cost = tail[0], tail[1]
        rem_seq = tail[2]

    assert total_cost == root_cost and total_purchase == root_used
    assert total_purchase <= budget
    assert periods[-1].remaining_budget == budget - total_purchase
    return Solution(True, tuple(periods), total_cost, total_purchase, soc,
                    purchase_budget=budget)


# ---------------------------------------------------------------------------
# Brute-force reference (test-only): enumerate every legal action sequence.
# ---------------------------------------------------------------------------

def brute_force(
    p: ScenarioParams, purchase_budget: Optional[int] = None
) -> Solution:
    """Reference solver that enumerates *all* legal action sequences.

    When ``purchase_budget`` is given, complete plans are admissible only if
    their total purchase respects it (the budget constrains the full tree
    search, not a filtered post-processing of a chosen plan).
    """
    n = p.n
    cap = p.capacity
    mc = p.max_charge
    budget = p.purchase_budget if purchase_budget is None else purchase_budget
    best: Optional[Label] = None
    best_periods: Optional[Tuple[PeriodResult, ...]] = None
    final_soc = p.initial_soc
    physically_feasible = False
    min_physical_purchase: Optional[int] = None

    def rec(t: int, soc: int, seq: List[Tuple[int, int]]) -> None:
        nonlocal best, best_periods, final_soc
        nonlocal physically_feasible, min_physical_purchase
        if t == n:
            if soc < p.terminal_min_soc:
                return
            periods: List[PeriodResult] = []
            total_cost = 0
            total_purchase = 0
            s = p.initial_soc
            for t2, ((c, d), net, price) in enumerate(zip(
                seq,
                [p.load[i] - p.pv[i] for i in range(n)],
                p.price,
            )):
                ns = s + c - d
                purchase = max(0, net - d + c)
                residual = net - d + c
                total_purchase += purchase
                total_cost += price * purchase
                s = ns
                periods.append(PeriodResult(
                    period=t2,
                    soc_start=None,  # replaced after budget check
                    charge=c,
                    discharge=d,
                    soc_end=ns,
                    net_load=net,
                    purchase=purchase,
                    curtail=max(0, -residual),
                    cost=price * purchase,
                ))
            physically_feasible = True
            if (min_physical_purchase is None
                    or total_purchase < min_physical_purchase):
                min_physical_purchase = total_purchase
            if budget is not None and total_purchase > budget:
                return
            # fill per-period evidence (soc_start and remaining budget)
            s = p.initial_soc
            used = 0
            filled: List[PeriodResult] = []
            for r in periods:
                used += r.purchase
                remain = None if budget is None else budget - used
                filled.append(PeriodResult(
                    period=r.period,
                    soc_start=s,
                    charge=r.charge,
                    discharge=r.discharge,
                    soc_end=r.soc_end,
                    net_load=r.net_load,
                    purchase=r.purchase,
                    curtail=r.curtail,
                    cost=r.cost,
                    remaining_budget=remain,
                ))
                s = r.soc_end
            flat = tuple(x for pair in seq for x in pair)
            label: Label = (total_cost, total_purchase, flat)
            if best is None or label < best:
                best = label
                best_periods = tuple(filled)
                final_soc = s
            return

        net = p.load[t] - p.pv[t]
        max_dis = min(mc, max(0, net), soc)
        for c in range(0, min(mc, cap - soc) + 1):
            for d in range(0, max_dis + 1):
                if c > 0 and d > 0:
                    continue  # no simultaneous charge and discharge
                seq.append((c, d))
                rec(t + 1, soc + c - d, seq)
                seq.pop()

    if not (0 <= p.initial_soc <= cap):
        return Solution(False, (), 0, 0, None,
                        "initial SoC outside [0, capacity]",
                        purchase_budget=budget)

    rec(0, p.initial_soc, [])
    if best is None:
        if physically_feasible:
            assert min_physical_purchase is not None
            return Solution(
                False, (), 0, 0, None,
                f"purchase budget {budget} is insufficient: every physically "
                f"feasible plan requires at least {min_physical_purchase} total purchase",
                purchase_budget=budget,
                minimum_purchase=min_physical_purchase,
            )
        return Solution(
            False, (), 0, 0, None,
            "cannot reach terminal minimum SoC within charge/discharge limits",
            purchase_budget=budget,
        )
    assert best_periods is not None
    return Solution(True, best_periods, best[0], best[1], final_soc,
                    purchase_budget=budget)


def normalize(
    load: Sequence[int],
    pv: Sequence[int],
    price: Sequence[int],
    capacity: int,
    initial_soc: int,
    terminal_min_soc: int,
    max_charge: int,
    purchase_budget: Optional[int] = None,
) -> ScenarioParams:
    return ScenarioParams(
        load=tuple(load),
        pv=tuple(pv),
        price=tuple(price),
        capacity=capacity,
        initial_soc=initial_soc,
        terminal_min_soc=terminal_min_soc,
        max_charge=max_charge,
        purchase_budget=purchase_budget,
    )
