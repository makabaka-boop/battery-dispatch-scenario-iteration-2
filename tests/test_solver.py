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
