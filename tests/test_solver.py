"""Exhaustive cross-checks of DP solver against a brute-force enumerator."""

import random

import pytest

from app.solver import (
    ScenarioParams,
    brute_force,
    normalize,
    solve,
)


def assert_invariants(p: ScenarioParams, sol) -> None:
    assert sol.feasible
    soc = p.initial_soc
    cum_purchase = 0
    for t, r in enumerate(sol.periods):
        assert r.soc_start == soc
        assert 0 <= soc <= p.capacity
        assert not (r.charge > 0 and r.discharge > 0), "same-period charge+discharge"
        assert 0 <= r.charge <= p.max_charge
        assert 0 <= r.discharge <= p.max_charge
        net = p.load[t] - p.pv[t]
        assert r.discharge <= max(0, net), "discharge exceeds positive net load"
        assert r.soc_end == soc + r.charge - r.discharge
        assert 0 <= r.soc_end <= p.capacity
        residual = net - r.discharge + r.charge
        assert r.purchase == max(0, residual)
        assert r.curtail == max(0, -residual)
        assert not (r.purchase > 0 and r.curtail > 0), "buying and selling together"
        assert r.cost == r.purchase * p.price[t]
        cum_purchase += r.purchase
        if p.purchase_budget is not None:
            # per-period evidence: remaining budget is cross-computable from
            # the budget and cumulative purchase, never negative.
            assert r.remaining_budget == p.purchase_budget - cum_purchase
            assert r.remaining_budget >= 0
        else:
            assert r.remaining_budget is None
        soc = r.soc_end
    assert soc >= p.terminal_min_soc
    assert soc == sol.final_soc
    assert sol.total_cost == sum(r.cost for r in sol.periods)
    assert sol.total_purchase == sum(r.purchase for r in sol.periods)
    assert sol.total_purchase == cum_purchase
    if p.purchase_budget is not None:
        assert sol.total_purchase <= p.purchase_budget
        assert sol.periods[-1].remaining_budget == (
            p.purchase_budget - sol.total_purchase
        )
        assert sol.purchase_budget == p.purchase_budget


def same_solution(a, b) -> bool:
    if a.feasible != b.feasible:
        return False
    if not a.feasible:
        # both infeasible must agree on *why* (physical vs budget) and on
        # the minimum purchase diagnostic; neither carries a partial plan
        if a.minimum_purchase != b.minimum_purchase:
            return False
        if bool(a.minimum_purchase is None) != bool(b.minimum_purchase is None):
            return False
        return not a.periods and not b.periods
    return (
        a.total_cost == b.total_cost
        and a.total_purchase == b.total_purchase
        and [(r.charge, r.discharge) for r in a.periods]
        == [(r.charge, r.discharge) for r in b.periods]
        and (
            a.purchase_budget != b.purchase_budget
            or [r.remaining_budget for r in a.periods]
            == [r.remaining_budget for r in b.periods]
        )
        and a.final_soc == b.final_soc
    )


def random_params(rng, min_n=1, max_n=4, purchase_budget="__absent__"):
    """Tiny scenarios so brute-force over *all* action trees stays cheap."""
    n = rng.randint(min_n, max_n)
    cap = rng.randint(1, 5)
    mc = rng.randint(1, cap)
    kwargs = dict(
        load=[rng.randint(0, 5) for _ in range(n)],
        pv=[rng.randint(0, 5) for _ in range(n)],
        price=[rng.randint(0, 6) for _ in range(n)],
        capacity=cap,
        initial_soc=rng.randint(0, cap),
        terminal_min_soc=rng.randint(0, cap),
        max_charge=mc,
    )
    if purchase_budget != "__absent__":
        kwargs["purchase_budget"] = purchase_budget
    return normalize(**kwargs)


@pytest.mark.parametrize("seed", range(300))
def test_dp_matches_bruteforce_random_short(seed):
    rng = random.Random(seed)
    p = random_params(rng)
    dp = solve(p)
    bf = brute_force(p)
    if dp.feasible:
        assert_invariants(p, dp)
    assert same_solution(dp, bf), (
        f"seed={seed} mismatch: dp=({dp.total_cost},{dp.total_purchase}) "
        f"bf=({bf.total_cost},{bf.total_purchase})"
    )


FIXED = [
    # all surplus PV, idle battery: everything curtailed, no purchase
    normalize([0, 0], [4, 5], [1, 1], 4, 0, 0, 2),
    # terminal unreachable: only n=2 charges of size 1 from SoC 0
    normalize([3, 3], [0, 0], [2, 2], 5, 0, 4, 1),
    # no PV, forced purchase
    normalize([3, 3, 3], [0, 0, 0], [1, 2, 3], 4, 0, 0, 2),
    # mc = 1, full battery
    normalize([2, 2, 2, 2], [0, 0, 0, 0], [5, 1, 1, 5], 3, 3, 3, 1),
    # zero prices everywhere -> cost objective mute, purchase decides, then lex
    normalize([0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6),
]


@pytest.mark.parametrize("p", FIXED)
def test_fixed_cases_against_bruteforce(p):
    dp, bf = solve(p), brute_force(p)
    assert same_solution(dp, bf)
    if dp.feasible:
        assert_invariants(p, dp)


def test_infeasible_reported():
    p = normalize([3, 3], [0, 0], [2, 2], 5, 0, 4, 1)
    sol = solve(p)
    assert not sol.feasible
    assert sol.periods == ()
    assert sol.final_soc is None
    assert "terminal" in sol.reason


def test_lexicographic_tiebreak():
    """Zero prices, more surplus than needed: several plans tie on cost and
    purchase; the lexicographically smallest charge/discharge sequence must
    delay charging (smallest c at the earliest period).

    nets [-4, -4, 6], cap 6: feasible zero-purchase plans charge
    (2,4,0), (3,3,0) or (4,2,0) before discharging 6. Smallest is (2,4,0).
    """
    p = normalize([0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6)
    sol = solve(p)
    assert sol.feasible
    assert sol.total_cost == 0
    assert sol.total_purchase == 0
    assert [r.charge for r in sol.periods] == [2, 4, 0]
    assert [r.discharge for r in sol.periods] == [0, 0, 6]
    assert [r.curtail for r in sol.periods] == [2, 0, 0]
    assert_invariants(p, sol)


def test_cost_before_purchase():
    """A tiny purchase at a high-priced shortage is cheaper than shifting
    more energy at an even higher price: cost dominates even if kWh grow."""
    # net [4, 0], cap 2, init 2, mc 2, prices [100, 1]
    # options: discharge 2 at t0 -> buy 2@100=200; idle -> buy 4@100=400
    p = normalize([4, 0], [0, 0], [100, 1], 2, 2, 0, 2)
    sol = solve(p)
    assert sol.feasible
    assert [r.discharge for r in sol.periods] == [2, 0]
    assert sol.total_cost == 200


def test_larger_random_invariants_no_bruteforce():
    """48 periods, capacity 50: DP only, check physical invariants."""
    rng = random.Random(4242)
    p = normalize(
        load=[rng.randint(0, 30) for _ in range(48)],
        pv=[rng.randint(0, 25) for _ in range(48)],
        price=[rng.randint(0, 20) for _ in range(48)],
        capacity=50,
        initial_soc=10,
        terminal_min_soc=10,
        max_charge=12,
    )
    sol = solve(p)
    assert sol.feasible
    assert_invariants(p, sol)


# --------------------------------------------------------------------------
# purchase budget: full legal-action-tree cross-checks at the budget boundary
# --------------------------------------------------------------------------

def _base_kwargs(rng):
    n = rng.randint(1, 4)
    cap = rng.randint(1, 5)
    return dict(
        load=[rng.randint(0, 5) for _ in range(n)],
        pv=[rng.randint(0, 5) for _ in range(n)],
        price=[rng.randint(0, 6) for _ in range(n)],
        capacity=cap,
        initial_soc=rng.randint(0, cap),
        terminal_min_soc=rng.randint(0, cap),
        max_charge=rng.randint(1, cap),
    )


@pytest.mark.parametrize("seed", range(300))
def test_budget_boundary_matches_bruteforce(seed):
    """For every random short scenario, cross-check both solvers at budgets
    below/at/above the unconstrained minimum purchase (the boundary) plus a
    very loose budget: feasible plans, diagnostics and remaining-budget
    evidence must all agree."""
    rng = random.Random(10_000 + seed)
    kw = _base_kwargs(rng)
    unb = solve(normalize(**kw))
    budgets = [0]
    if unb.feasible:
        q = unb.total_purchase
        budgets += [max(0, q - 1), q, q + 1, q + 5]
    for b in budgets:
        p = normalize(purchase_budget=b, **kw)
        dp, bf = solve(p), brute_force(p)
        assert same_solution(dp, bf), (
            f"seed={seed} B={b} mismatch: "
            f"dp=({dp.feasible},{dp.total_cost},{dp.total_purchase}) "
            f"bf=({bf.feasible},{bf.total_cost},{bf.total_purchase})"
        )
        if dp.feasible:
            assert_invariants(p, dp)
            assert dp.total_purchase <= b
        else:
            assert dp.periods == () and bf.periods == ()
            if unb.feasible:
                # physically feasible but budget too tight -> budget failure
                assert dp.minimum_purchase == q
                assert b < q
            else:
                # physical infeasibility is not misreported as budget failure
                assert dp.minimum_purchase is None


@pytest.mark.parametrize("seed", range(200))
def test_budget_wide_random_matches_bruteforce(seed):
    """Random budget values (not just boundary) across full action trees."""
    rng = random.Random(20_000 + seed)
    kw = _base_kwargs(rng)
    for b in (rng.choice([0, 1, 2, 3, 5, 9, 20]) for _ in range(3)):
        p = normalize(purchase_budget=b, **kw)
        dp, bf = solve(p), brute_force(p)
        assert same_solution(dp, bf)
        if dp.feasible:
            assert_invariants(p, dp)


def test_budget_boundary_exact():
    """A fixed two-period case pinned around its boundary.

    nets [4, 4], battery starts full at 2.  Every plan buys at least 6:
    discharging 2 at t0 buys (2, 4), discharging 2 at t1 buys (4, 2).
    Budget 6 is feasible; budget 5 fails with minimum_purchase == 6.
    """
    kw = dict(load=[4, 4], pv=[0, 0], price=[100, 1],
              capacity=2, initial_soc=2, terminal_min_soc=0, max_charge=2)

    tight = solve(normalize(purchase_budget=6, **kw))
    assert tight.feasible
    assert tight.total_purchase == 6
    assert tight.total_purchase <= 6
    assert [r.remaining_budget for r in tight.periods] == [4, 0]

    short = solve(normalize(purchase_budget=5, **kw))
    assert not short.feasible
    assert short.periods == ()
    assert short.final_soc is None
    assert short.minimum_purchase == 6
    assert short.purchase_budget == 5
    assert "budget" in short.reason


def test_cost_beats_purchase_under_binding_budget():
    """费用与用量目标冲突：at the binding budget two plans use the SAME
    minimum quantity 6 but different cost; the cheaper (discharge at price
    100 to avoid buying at price 100 vs shifting to price 1) must win.

    q=6 plans: discharge 2 at t0 -> cost 2*100 + 4*1 = 204;
               discharge 2 at t1 -> cost 4*100 + 2*1 = 402.
    """
    kw = dict(load=[4, 4], pv=[0, 0], price=[100, 1],
              capacity=2, initial_soc=2, terminal_min_soc=0, max_charge=2)
    sol = solve(normalize(purchase_budget=6, **kw))
    assert sol.feasible
    assert sol.total_purchase == 6
    assert sol.total_cost == 204
    assert [r.discharge for r in sol.periods] == [2, 0]
    # a non-binding budget must not change the lexicographic optimum either
    loose = solve(normalize(purchase_budget=7, **kw))
    assert same_solution(loose, sol)


def test_lex_tiebreak_under_budget():
    """并列裁决：zero prices mute the cost objective; at budget 0 several
    zero-purchase plans tie on cost and quantity, so the lexicographically
    smallest charge/discharge sequence wins (unchanged from unconstrained)."""
    p = normalize([0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6,
                  purchase_budget=0)
    sol = solve(p)
    assert sol.feasible
    assert sol.total_cost == 0
    assert sol.total_purchase == 0
    assert [r.charge for r in sol.periods] == [2, 4, 0]
    assert [r.discharge for r in sol.periods] == [0, 0, 6]
    assert [r.remaining_budget for r in sol.periods] == [0, 0, 0]
    assert_invariants(p, sol)
    # omitting the budget yields the identical plan
    assert same_solution(sol, solve(normalize(
        [0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6)))


def test_budget_shortfall_distinct_from_physical_infeasible():
    """两种无解必须可区分：a physically reachable case failing only on budget
    carries minimum_purchase; a physically unreachable case never does, even
    when a generous budget is given.  Neither response has a partial plan."""
    # budget-only failure
    short = solve(normalize(
        [4, 4], [0, 0], [1, 1], 2, 2, 0, 2, purchase_budget=0))
    assert not short.feasible
    assert short.minimum_purchase == 6
    assert short.periods == ()
    assert short.final_soc is None

    # physical failure independent of budget
    phys = solve(normalize(
        [3, 3], [0, 0], [2, 2], 5, 0, 4, 1, purchase_budget=1_000_000))
    assert not phys.feasible
    assert phys.minimum_purchase is None
    assert "terminal" in phys.reason
    assert phys.periods == ()
    assert phys.final_soc is None


def test_budget_zero_with_pv_only_feasible():
    """All-surplus scenario is feasible at budget 0 and buys nothing."""
    p = normalize([0, 0], [4, 5], [1, 1], 4, 0, 0, 2, purchase_budget=0)
    sol = solve(p)
    assert sol.feasible
    assert sol.total_purchase == 0
    assert all(r.remaining_budget == 0 for r in sol.periods)
    assert_invariants(p, sol)


def test_budget_does_not_postfilter_unconstrained_plan():
    """The budgeted optimum is the genuine constrained optimum, never an
    unconstrained plan salvaged by filtering: brute-force agrees at every
    budget, and a tight budget infeasibility is reported rather than a
    trimmed partial plan."""
    kw = dict(load=[5, 1], pv=[0, 0], price=[10, 10],
              capacity=3, initial_soc=3, terminal_min_soc=0, max_charge=3)
    for b in range(0, 8):
        p = normalize(purchase_budget=b, **kw)
        dp, bf = solve(p), brute_force(p)
        assert same_solution(dp, bf), b
        if dp.feasible:
            assert_invariants(p, dp)
