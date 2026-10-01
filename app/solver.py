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

Battery capacity is at most 50 (enforced by the API model), so exhaustive
dynamic programming over integer states is trivially fast and exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class Solution:
    feasible: bool
    periods: Tuple[PeriodResult, ...]
    total_cost: int
    total_purchase: int
    final_soc: Optional[int]
    reason: Optional[str] = None


# (cost, purchase, sequence-tuple) is already the lexicographic order of the
# three objectives, so Python tuple comparison implements the tie break.
Label = Tuple[int, int, Tuple[int, ...]]


def _better(a: Optional[Label], b: Optional[Label]) -> Optional[Label]:
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def solve(p: ScenarioParams) -> Solution:
    """Exact backward DP with lexicographic action reconstruction."""
    n = p.n
    cap = p.capacity
    mc = p.max_charge

    if not (0 <= p.initial_soc <= cap):
        return Solution(False, (), 0, 0, None, "initial SoC outside [0, capacity]")

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

    root = dp[0][p.initial_soc]
    if root is None:
        return Solution(False, (), 0, 0, None,
                        "cannot reach terminal minimum SoC within charge/discharge limits")

    # Forward reconstruction: choose the lexicographically smallest feasible
    # action at each state (the backward suffixes already carry optimal labels,
    # so greedily picking the smallest action that keeps the optimum is valid).
    actions: List[Tuple[int, int]] = []
    soc = p.initial_soc
    remaining_cost = root[0]
    remaining_purchase = root[1]
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
        ))
        total_cost += cost
        total_purchase += purchase
        soc = ns
        remaining_cost, remaining_purchase = tail[0], tail[1]

    assert total_cost == root[0] and total_purchase == root[1]
    return Solution(True, tuple(periods), total_cost, total_purchase, soc)


# ---------------------------------------------------------------------------
# Brute-force reference (test-only): enumerate every legal action sequence.
# ---------------------------------------------------------------------------

def brute_force(p: ScenarioParams) -> Solution:
    """Reference solver that enumerates *all* legal action sequences."""
    n = p.n
    cap = p.capacity
    mc = p.max_charge
    best: Optional[Label] = None
    best_periods: Optional[Tuple[PeriodResult, ...]] = None
    final_soc = p.initial_soc

    def rec(t: int, soc: int, seq: List[Tuple[int, int]]) -> None:
        nonlocal best, best_periods, final_soc
        if t == n:
            if soc < p.terminal_min_soc:
                return
            periods: List[PeriodResult] = []
            total_cost = 0
            total_purchase = 0
            s = p.initial_soc
            for (c, d), net, price in zip(
                seq,
                [p.load[i] - p.pv[i] for i in range(n)],
                p.price,
            ):
                ns = s + c - d
                purchase = max(0, net - d + c)
                residual = net - d + c
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
                ))
                total_cost += price * purchase
                total_purchase += purchase
                s = ns
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
        return Solution(False, (), 0, 0, None,
                        "cannot reach terminal minimum SoC within charge/discharge limits")
    assert best_periods is not None
    return Solution(True, best_periods, best[0], best[1], final_soc)


def normalize(
    load: Sequence[int],
    pv: Sequence[int],
    price: Sequence[int],
    capacity: int,
    initial_soc: int,
    terminal_min_soc: int,
    max_charge: int,
) -> ScenarioParams:
    return ScenarioParams(
        load=tuple(load),
        pv=tuple(pv),
        price=tuple(price),
        capacity=capacity,
        initial_soc=initial_soc,
        terminal_min_soc=terminal_min_soc,
        max_charge=max_charge,
    )
