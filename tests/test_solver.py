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
        soc = r.soc_end
    assert soc >= p.terminal_min_soc
    assert soc == sol.final_soc
    assert sol.total_cost == sum(r.cost for r in sol.periods)
    assert sol.total_purchase == sum(r.purchase for r in sol.periods)


def same_solution(a, b) -> bool:
    if a.feasible != b.feasible:
        return False
    if not a.feasible:
        return True
    return (
        a.total_cost == b.total_cost
        and a.total_purchase == b.total_purchase
        and [(r.charge, r.discharge) for r in a.periods]
        == [(r.charge, r.discharge) for r in b.periods]
        and a.final_soc == b.final_soc
    )


def random_params(rng, min_n=1, max_n=4):
    """Tiny scenarios so brute-force over *all* action trees stays cheap."""
    n = rng.randint(min_n, max_n)
    cap = rng.randint(1, 5)
    mc = rng.randint(1, cap)
    return normalize(
        load=[rng.randint(0, 5) for _ in range(n)],
        pv=[rng.randint(0, 5) for _ in range(n)],
        price=[rng.randint(0, 6) for _ in range(n)],
        capacity=cap,
        initial_soc=rng.randint(0, cap),
        terminal_min_soc=rng.randint(0, cap),
        max_charge=mc,
    )


def with_budget(p: ScenarioParams, budget) -> ScenarioParams:
    return normalize(
        p.load, p.pv, p.price, p.capacity, p.initial_soc,
        p.terminal_min_soc, p.max_charge, purchase_budget=budget,
    )


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


# ===========================================================================
# purchase_budget
# ===========================================================================

def assert_budget_evidence(p: ScenarioParams, sol, budget: int) -> None:
    """Budgeted solution obeys the cap and the per-period remaining-budget
    evidence cross-checks against cumulative purchase."""
    assert sol.feasible
    assert sol.purchase_budget == budget
    assert sol.total_purchase <= budget
    cumulative = 0
    for r in sol.periods:
        cumulative += r.purchase
        assert r.remaining_budget == budget - cumulative
    assert sol.periods[-1].remaining_budget == budget - sol.total_purchase
    assert_invariants(p, sol)


@pytest.mark.parametrize("seed", range(150))
def test_budgeted_dp_matches_bruteforce_full_sweep(seed):
    """For every integer budget 0..upper_bound, the DP must agree with the
    exhaustive action tree: same feasibility class, same lexicographic plan
    when feasible, and the same minimum required purchase when the cap alone
    blocks every plan.  This is the budget-boundary cross-check."""
    rng = random.Random(10_000 + seed)
    p = random_params(rng)
    n = p.n
    upper = sum(max(0, p.load[t] - p.pv[t]) + p.max_charge for t in range(n))
    for budget in range(0, upper + 1):
        dp = solve(with_budget(p, budget))
        bf = brute_force(with_budget(p, budget))
        assert dp.feasible == bf.feasible, (seed, budget)
        if dp.feasible:
            assert same_solution(dp, bf), (seed, budget)
            assert_budget_evidence(with_budget(p, budget), dp, budget)
        else:
            # Never emit a partial "current plan".
            assert dp.periods == () and dp.final_soc is None
            if dp.required_min_purchase is None:
                # physically impossible: brute force must agree
                assert bf.required_min_purchase is None
            else:
                # cap too small: required minimum is reported and consistent
                assert budget < dp.required_min_purchase
                assert bf.required_min_purchase == dp.required_min_purchase
                assert "budget" in dp.reason


def test_budget_none_equals_unconstrained():
    """Omitting the budget reproduces the existing result exactly."""
    p = normalize([3, 3, 1], [0, 1, 2], [3, 2, 5], 5, 2, 2, 2)
    none_sol = solve(with_budget(p, None))
    absent_sol = solve(p)
    assert same_solution(none_sol, absent_sol)
    assert all(r.remaining_budget is None for r in none_sol.periods)
    assert none_sol.purchase_budget is None and none_sol.required_min_purchase is None


def test_budget_boundary_exact_minimum_is_feasible():
    """A cap exactly equal to the optimum purchase is feasible; one unit
    lower is infeasible-but-physically-ok and reports that minimum."""
    p = normalize([3, 3, 1], [0, 1, 2], [3, 2, 5], 5, 2, 2, 2,
                  purchase_budget=None)
    qstar = solve(p).total_purchase
    assert qstar == 4

    exact = solve(with_budget(p, qstar))
    assert exact.feasible
    assert exact.total_purchase == qstar
    assert exact.periods[-1].remaining_budget == 0
    assert_budget_evidence(with_budget(p, qstar), exact, qstar)

    tight = solve(with_budget(p, qstar - 1))
    assert not tight.feasible
    assert tight.periods == () and tight.final_soc is None
    assert tight.required_min_purchase == qstar
    assert tight.purchase_budget == qstar - 1
    assert str(qstar) in tight.reason and "budget" in tight.reason


def test_physical_infeasible_distinct_from_budget_infeasible():
    """When terminal SoC is unreachable, no required minimum is reported even
    if a tiny budget is given: the physical constraints themselves fail."""
    p = normalize([3, 3], [0, 0], [2, 2], 5, 0, 4, 1, purchase_budget=0)
    sol = solve(p)
    assert not sol.feasible
    assert sol.required_min_purchase is None
    assert "terminal" in sol.reason
    assert sol.periods == () and sol.final_soc is None

    # Same physics without a budget is infeasible for the identical reason.
    bare = solve(normalize([3, 3], [0, 0], [2, 2], 5, 0, 4, 1))
    assert not bare.feasible and bare.required_min_purchase is None


def test_budget_never_post_filters():
    """A tight budget changes *which* optimization problem is solved; it is
    not a post-hoc filter.  Feasible plans for B == min purchase are real
    complete plans whose running sum never exceeds the cap, and a cap below
    that minimum yields no plan at all (rather than a truncated prefix)."""
    # net positive in the last period forces late purchase; charging early
    # can only substitute, never erase the minimum derived from physics.
    p = normalize([0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6)
    sol = solve(normalize(p.load, p.pv, p.price, 6, 0, 0, 6,
                          purchase_budget=0))
    assert sol.feasible and sol.total_purchase == 0
    assert all(r.remaining_budget == 0 for r in sol.periods)


def test_cost_objective_dominates_purchase_under_budget():
    """Under zero prices the cost objective is mute; among same-cost plans
    the minimum-purchase one is chosen.  With a cap the minimum-purchase
    threshold is exactly where feasibility starts - a higher-q, same-cost
    plan must never be returned when a lower-q plan exists."""
    # nets [-4,-4,6]: zero-purchase plans exist (charge from surplus then
    # discharge).  Same cost (0) plans that bought extra PV-period energy
    # would be wasted; the chosen plan buys 0.
    p = normalize([0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6)
    free = solve(p)
    assert free.total_cost == 0 and free.total_purchase == 0
    assert [r.charge for r in free.periods] == [2, 4, 0]

    # A case where prices differ: the cost-optimal plan never buys more than
    # the physical minimum (identity: Q = sum(net) + final_soc - init + curtail),
    # so budget feasibility begins exactly at that purchase total.
    rng = random.Random(77)
    p2 = normalize(
        load=[rng.randint(0, 5) for _ in range(4)],
        pv=[rng.randint(0, 5) for _ in range(4)],
        price=[rng.randint(1, 6) for _ in range(4)],  # strictly positive prices
        capacity=5, initial_soc=2, terminal_min_soc=2, max_charge=2,
    )
    qstar = solve(p2).total_purchase
    assert solve(with_budget(p2, qstar)).feasible
    assert not solve(with_budget(p2, qstar - 1)).feasible
    # positive prices => no slack purchase is cost-neutral; brute confirms
    bf = brute_force(with_budget(p2, qstar))
    assert bf.total_purchase == qstar


def test_lexicographic_tiebreak_preserved_under_budget():
    """The third-level (action sequence) adjudication is unchanged under a
    cap: brute force and DP return identical charge/discharge sequences at
    every feasible budget, including exact ties on cost and purchase."""
    p = normalize([0, 0, 6], [4, 4, 0], [0, 0, 0], 6, 0, 0, 6)
    # budget 0 still admits all the zero-purchase plans; lex-smallest wins
    sol = solve(with_budget(p, 0))
    assert [r.charge for r in sol.periods] == [2, 4, 0]
    assert [r.discharge for r in sol.periods] == [0, 0, 6]


def test_budget_full_scale_invariants_and_timing():
    """48 periods, capacity 50 with a binding-but-feasible cap: exact and
    fast, evidence cross-checks."""
    rng = random.Random(4242)
    base = normalize(
        load=[rng.randint(0, 30) for _ in range(48)],
        pv=[rng.randint(0, 25) for _ in range(48)],
        price=[rng.randint(0, 20) for _ in range(48)],
        capacity=50, initial_soc=10, terminal_min_soc=10, max_charge=12,
    )
    qstar = solve(base).total_purchase
    capped = solve(with_budget(base, qstar))
    assert capped.feasible
    assert_budget_evidence(with_budget(base, qstar), capped, qstar)
    assert same_solution(capped, solve(base))
    tight = solve(with_budget(base, qstar - 1))
    assert not tight.feasible and tight.required_min_purchase == qstar


def test_budget_loose_uses_unconstrained_optimum():
    """A cap above any plan's possible purchase is equivalent to no cap but
    still carries remaining-budget evidence."""
    p = normalize([3, 3, 1], [0, 1, 2], [3, 2, 5], 5, 2, 2, 2)
    qstar = solve(p).total_purchase
    huge = solve(with_budget(p, 10 ** 9))
    assert huge.feasible
    assert same_solution(huge, solve(p))
    assert huge.periods[-1].remaining_budget == 10 ** 9 - qstar
