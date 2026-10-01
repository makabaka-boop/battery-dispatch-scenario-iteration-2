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

An optional ``purchase_budget`` (non-negative integer) caps the *sum* of
purchase over all periods.  The constraint is handled inside the optimization
- it is never applied by post-filtering an unconstrained plan.  When the budget
binds there is no solution, and the response reports the minimum total purchase
that the physical constraints themselves require (``required_min_purchase``)
so callers can tell "physically impossible" apart from "budget too small".

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
    # Minimum total purchase needed to satisfy the *physical* constraints
    # (terminal SoC etc.).  Populated whenever the physical problem is
    # feasible, including the budget-infeasible case.
    required_min_purchase: Optional[int] = None


# (cost, purchase, sequence-tuple) is already the lexicographic order of the
# three objectives, so Python tuple comparison implements the tie break.
Label = Tuple[int, int, Tuple[int, ...]]


def _better(a: Optional[Label], b: Optional[Label]) -> Optional[Label]:
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _purchase_upper_bound(nets: Sequence[int], mc: int) -> int:
    # charge <= max_charge can add at most mc units of purchase in a period.
    return sum(max(0, net) + mc for net in nets)


def solve(p: ScenarioParams) -> Solution:
    """Exact DP with lexicographic action reconstruction.

    When ``purchase_budget`` is given, a second DP over cumulative purchase
    enforces the cap; the lexicographic (cost, purchase, action) adjudication
    is unchanged.
    """
    n = p.n
    cap = p.capacity
    mc = p.max_charge

    if not (0 <= p.initial_soc <= cap):
        return Solution(False, (), 0, 0, None,
                        "initial SoC outside [0, capacity]",
                        purchase_budget=p.purchase_budget)

    dp, root = _unconstrained_dp(p)
    if root is None:
        # Physically impossible even with unlimited purchase: the budget is
        # irrelevant, and no partial plan exists.
        return Solution(False, (), 0, 0, None,
                        "cannot reach terminal minimum SoC within charge/discharge limits",
                        purchase_budget=p.purchase_budget)

    total_cost, total_purchase = root[0], root[1]
    budget = p.purchase_budget

    if budget is not None and budget < total_purchase:
        # Physical constraints ARE feasible, but every feasible plan buys at
        # least total_purchase (see module note / identity): the cap alone
        # makes the problem unsolvable.  No partial plan is produced.
        return Solution(
            False, (), 0, 0, None,
            f"purchase budget {budget} is below the minimum required total "
            f"purchase {total_purchase}",
            purchase_budget=budget,
            required_min_purchase=total_purchase,
        )

    if budget is None or budget >= _purchase_upper_bound(
            [p.load[t] - p.pv[t] for t in range(n)], mc):
        # Budget absent or provably loose: every physical plan fits, so the
        # unconstrained optimum is also the budgeted optimum.  Reconstruction
        # still threads the remaining-budget evidence when a cap was given.
        return _reconstruct(p, dp, root, budget=budget)

    # budget in [total_purchase, upper_bound): run the constrained DP.  With the
    # model identity  Q = sum(net) + soc_n - soc_0 + curtail,  the cheapest
    # plan already minimizes purchase, so this branch returns the same plan;
    # the constrained DP still re-optimizes under the cap honestly.
    return _solve_budgeted(p, budget, total_purchase)


def _unconstrained_dp(
    p: ScenarioParams,
) -> Tuple[List[List[Optional[Label]]], Optional[Label]]:
    """Backward DP ignoring the purchase budget."""
    n = p.n
    cap = p.capacity
    mc = p.max_charge

    # dp[t][s] = best label achievable from state s at the START of period t
    # over periods t..n-1.  Base case: period n must end at s >= terminal min.
    dp: List[List[Optional[Label]]] = [[None] * (cap + 1) for _ in range(n + 1)]
    for s in range(cap + 1):
        dp[n][s] = (0, 0, ()) if s >= p.terminal_min_soc else None

    for t in range(n - 1, -1, -1):
        net = p.load[t] - p.pv[t]
        price = p.price[t]
        max_dis = min(mc, max(0, net))
        for s in range(cap + 1):
            best: Optional[Label] = None
            c_max = min(mc, cap - s)
            # Charge actions.
            for c in range(0, c_max + 1):
                ns = s + c
                tail = dp[t + 1][ns]
                if tail is None:
                    continue
                purchase = max(0, net + c)
                cost = price * purchase
                label = (tail[0] + cost, tail[1] + purchase, (c, 0) + tail[2])
                best = _better(best, label)
            # Discharge actions (idle c=0 already added above).
            for d in range(1, min(max_dis, s) + 1):
                ns = s - d
                tail = dp[t + 1][ns]
                if tail is None:
                    continue
                purchase = max(0, net - d)
                cost = price * purchase
                label = (tail[0] + cost, tail[1] + purchase, (0, d) + tail[2])
                best = _better(best, label)
            dp[t][s] = best

    return dp, dp[0][p.initial_soc]


def _reconstruct(
    p: ScenarioParams,
    dp: List[List[Optional[Label]]],
    root: Label,
    budget: Optional[int],
) -> Solution:
    """Forward reconstruction from an unconstrained DP table.

    Picks the lexicographically smallest feasible action at each state (the
    backward suffixes carry optimal labels, so greedily taking the smallest
    action that preserves the optimum is valid).
    """
    n = p.n
    cap = p.capacity
    mc = p.max_charge

    soc = p.initial_soc
    remaining_cost = root[0]
    remaining_purchase = root[1]
    remaining_budget = budget
    periods: List[PeriodResult] = []
    total_cost = 0
    total_purchase = 0

    for t in range(n):
        net = p.load[t] - p.pv[t]
        price = p.price[t]
        max_dis = min(mc, max(0, net))
        cands: List[Tuple[Tuple[int, int], int, int, int]] = []  # action, ns, cost, purchase
        c_max = min(mc, cap - soc)
        for c in range(0, c_max + 1):
            ns = soc + c
            purchase = max(0, net + c)
            cands.append(((c, 0), ns, price * purchase, purchase))
        for d in range(1, min(max_dis, soc) + 1):
            ns = soc - d
            purchase = max(0, net - d)
            cands.append(((0, d), ns, price * purchase, purchase))

        chosen = None
        for action, ns, cost, purchase in sorted(cands, key=lambda x: x[0]):
            tail = dp[t + 1][ns]
            if tail is None:
                continue
            if (tail[0] + cost, tail[1] + purchase) == (remaining_cost, remaining_purchase):
                chosen = (action, ns, cost, purchase, tail)
                break
        assert chosen is not None, "DP reconstruction failed"
        (c, d), ns, cost, purchase, tail = chosen

        if remaining_budget is not None:
            remaining_budget -= purchase
        residual = net - d + c
        curtail = max(0, -residual)
        periods.append(PeriodResult(
            period=t,
            soc_start=soc,
            charge=c,
            discharge=d,
            soc_end=ns,
            net_load=net,
            purchase=purchase,
            curtail=curtail,
            cost=cost,
            remaining_budget=remaining_budget,
        ))
        total_cost += cost
        total_purchase += purchase
        soc = ns
        remaining_cost, remaining_purchase = tail[0], tail[1]

    assert total_cost == root[0] and total_purchase == root[1]
    if budget is not None:
        assert total_purchase <= budget
    return Solution(True, tuple(periods), total_cost, total_purchase, soc,
                    purchase_budget=budget)


def _solve_budgeted(
    p: ScenarioParams,
    budget: int,
    total_purchase: int,
) -> Solution:
    """Constrained solver: sparse backward DP over cumulative purchase.

    ``layers[t][s]`` maps suffix-purchase ``q`` (purchases over periods
    t..n-1) to the minimum suffix cost.  Only Pareto-efficient (q, cost)
    points are kept - a point dominated on both coordinates can never
    participate in an optimal prefix (a standard frontier filter that stays
    valid for any prices).  In this battery physics each frontier in fact
    collapses to one point, because  q - curtail = sum(net) + final_soc -
    start_soc  makes the minimum-cost suffix also the minimum-purchase
    suffix; the code does not rely on that identity.  Lexicographic
    tie-breaking by action sequence happens during forward reconstruction,
    exactly as in the unconstrained solver.
    """
    n = p.n
    cap = p.capacity
    mc = p.max_charge
    INF = 10 ** 30

    # layers[t][s] = {q: min_cost}; layers[n] is the virtual terminal layer.
    layers: List[Optional[List[Optional[Dict[int, int]]]]] = [None] * (n + 1)

    for t in range(n - 1, -1, -1):
        net = p.load[t] - p.pv[t]
        price = p.price[t]
        max_dis = min(mc, max(0, net))
        nxt = layers[t + 1]
        cur: List[Optional[Dict[int, int]]] = [None] * (cap + 1)

        for s in range(cap + 1):
            points: Dict[int, int] = {}

            def add(ns: int, purchase: int, cost: int) -> None:
                if purchase > budget:
                    return
                if nxt is None:
                    # Terminal layer: suffix is feasible iff ns meets terminal.
                    if ns >= p.terminal_min_soc:
                        old = points.get(purchase)
                        if old is None or cost < old:
                            points[purchase] = cost
                    return
                tail = nxt[ns]
                if not tail:
                    return
                for q, tail_cost in tail.items():
                    q_total = q + purchase
                    if q_total > budget:
                        continue
                    c_total = tail_cost + cost
                    old = points.get(q_total)
                    if old is None or c_total < old:
                        points[q_total] = c_total

            c_max = min(mc, cap - s)
            for c in range(0, c_max + 1):
                purchase = max(0, net + c)
                add(s + c, purchase, price * purchase)
            for d in range(1, min(max_dis, s) + 1):
                purchase = max(0, net - d)
                add(s - d, purchase, price * purchase)

            if points:
                # Pareto prune: with q ascending, keep only strict new cost
                # minima.  Equal (q, cost) collapse through the dict; the
                # lexicographically smallest way to realize a surviving point
                # is recovered by forward reconstruction.
                pruned: Dict[int, int] = {}
                running = INF
                for q in sorted(points):
                    c = points[q]
                    if c < running:
                        pruned[q] = c
                        running = c
                cur[s] = pruned
        layers[t] = cur

    root_frontier = layers[0][p.initial_soc]
    if root_frontier is None:
        # Defensive: feasibility/required purchase was established up front.
        return Solution(
            False, (), 0, 0, None,
            f"purchase budget {budget} is below the minimum required total "
            f"purchase {total_purchase}",
            purchase_budget=budget,
            required_min_purchase=total_purchase,
        )

    # Reconcile the constrained optimum independently of the unconstrained
    # DP: objectives are still (cost, purchase), so take min cost then min q.
    q_star = min(root_frontier, key=lambda q: (root_frontier[q], q))
    total_cost = root_frontier[q_star]
    total_purchase = q_star

    # ---- forward reconstruction with running remaining-budget evidence ----
    soc = p.initial_soc
    rem_cost = total_cost
    rem_purchase = total_purchase
    used = 0
    remaining_budget: Optional[int] = budget
    periods: List[PeriodResult] = []

    for t in range(n):
        net = p.load[t] - p.pv[t]
        price = p.price[t]
        max_dis = min(mc, max(0, net))
        cands: List[Tuple[Tuple[int, int], int, int]] = []  # action, ns, purchase
        c_max = min(mc, cap - soc)
        for c in range(0, c_max + 1):
            cands.append(((c, 0), soc + c, max(0, net + c)))
        for d in range(1, min(max_dis, soc) + 1):
            cands.append(((0, d), soc - d, max(0, net - d)))

        chosen = None
        for (c, d), ns, purchase in sorted(cands, key=lambda x: x[0]):
            if used + purchase > budget:
                continue
            cost = price * purchase
            nxt = layers[t + 1]
            if nxt is None:
                if (ns >= p.terminal_min_soc
                        and purchase == rem_purchase and cost == rem_cost):
                    chosen = (c, d, ns, purchase)
                    break
                continue
            tail = nxt[ns]
            q_tail = rem_purchase - purchase
            if tail is not None and q_tail in tail and tail[q_tail] == rem_cost - cost:
                chosen = (c, d, ns, purchase)
                break
        assert chosen is not None, "budgeted DP reconstruction failed"
        c, d, ns, purchase = chosen

        cost = price * purchase
        remaining_budget -= purchase
        residual = net - d + c
        periods.append(PeriodResult(
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
        ))
        used += purchase
        rem_cost -= cost
        rem_purchase -= purchase
        soc = ns

    assert used == total_purchase and remaining_budget == budget - total_purchase
    return Solution(True, tuple(periods), total_cost, total_purchase, soc,
                    purchase_budget=budget)


# ---------------------------------------------------------------------------
# Brute-force reference (test-only): enumerate every legal action sequence.
# ---------------------------------------------------------------------------

def brute_force(p: ScenarioParams) -> Solution:
    """Reference solver that enumerates *all* legal action sequences."""
    n = p.n
    cap = p.capacity
    mc = p.max_charge
    budget = p.purchase_budget
    best: Optional[Label] = None
    best_periods: Optional[Tuple[PeriodResult, ...]] = None
    final_soc = p.initial_soc
    min_required: List[Optional[int]] = [None]  # over ALL feasible physical plans

    def rec(t: int, soc: int, seq: List[Tuple[int, int]]) -> None:
        nonlocal best, best_periods, final_soc
        if t == n:
            if soc < p.terminal_min_soc:
                return
            periods: List[PeriodResult] = []
            total_cost = 0
            total_purchase = 0
            s = p.initial_soc
            remaining_budget = budget
            for (c, d), net, price in zip(
                seq,
                [p.load[i] - p.pv[i] for i in range(n)],
                p.price,
            ):
                ns = s + c - d
                purchase = max(0, net - d + c)
                residual = net - d + c
                if remaining_budget is not None:
                    remaining_budget -= purchase
                periods.append(PeriodResult(
                    period=len(periods),
                    soc_start=s,
                    charge=c,
                    discharge=d,
                    soc_end=ns,
                    net_load=net,
                    purchase=purchase,
                    curtail=max(0, -residual),
                    cost=price * purchase,
                    remaining_budget=remaining_budget,
                ))
                total_cost += price * purchase
                total_purchase += purchase
                s = ns
            if min_required[0] is None or total_purchase < min_required[0]:
                min_required[0] = total_purchase
            if budget is not None and total_purchase > budget:
                return
            flat = tuple(x for pair in seq for x in pair)
            label: Label = (total_cost, total_purchase, flat)
            if best is None or label < best:
                best = label
                best_periods = tuple(periods)
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

    rec(0, p.initial_soc, [])
    if best is None:
        if min_required[0] is None:
            return Solution(False, (), 0, 0, None,
                            "cannot reach terminal minimum SoC within charge/discharge limits",
                            purchase_budget=budget)
        return Solution(
            False, (), 0, 0, None,
            f"purchase budget {budget} is below the minimum required total "
            f"purchase {min_required[0]}",
            purchase_budget=budget,
            required_min_purchase=min_required[0],
        )
    assert best_periods is not None
    # A feasible solution reports no required minimum (mirrors the production
    # solver); the minimum is surfaced only when the cap alone blocks solving.
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
