"""API tests: revision optimism, stale-solve semantics, concurrency races."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.storage import RevisionConflict, ScenarioStore


@pytest.fixture()
def client(tmp_path):
    store = ScenarioStore(str(tmp_path / "db.json"))
    app = create_app(store)
    with TestClient(app) as c:
        yield c, store


def body_factory(**over):
    base = dict(
        name="case",
        load=[3] * 8,
        pv=[0, 0, 1, 4, 6, 3, 1, 0],
        price=[3, 2, 2, 4, 5, 6, 4, 3],
        capacity=20,
        initial_soc=5,
        terminal_min_soc=5,
        max_charge=5,
    )
    base.update(over)
    return base


def create(client, **over):
    c, _ = client
    r = c.post("/api/scenarios", json=body_factory(**over))
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_validation_periods(client):
    c, _ = client
    r = c.post("/api/scenarios", json=body_factory(load=[1] * 7, pv=[0] * 7, price=[1] * 7))
    assert r.status_code == 422
    r = c.post("/api/scenarios", json=body_factory(load=[1] * 49, pv=[0] * 49, price=[1] * 49))
    assert r.status_code == 422


def test_validation_capacity_and_negatives(client):
    c, _ = client
    assert c.post("/api/scenarios", json=body_factory(capacity=51)).status_code == 422
    bad = body_factory(pv=[-1] + [0] * 7)
    assert c.post("/api/scenarios", json=bad).status_code == 422
    assert c.post("/api/scenarios", json=body_factory(initial_soc=21)).status_code == 422
    assert c.post("/api/scenarios", json=body_factory(max_charge=21)).status_code == 422
    assert c.post("/api/scenarios", json=body_factory(terminal_min_soc=21)).status_code == 422
    assert c.post("/api/scenarios", json=body_factory(name="   ")).status_code == 422


def test_validation_length_mismatch(client):
    c, _ = client
    r = c.post("/api/scenarios", json=body_factory(pv=[0] * 7))
    assert r.status_code == 422


# --------------------------------------------------------------------------
# revision optimism
# --------------------------------------------------------------------------

def test_create_starts_at_revision_1(client):
    e = create(client)
    assert e["revision"] == 1
    assert e["id"]


def test_save_advances_revision_and_keeps_history(client):
    c, _ = client
    e = create(client)
    r = c.put(f"/api/scenarios/{e['id']}",
              json={"expected_revision": 1, "scenario": body_factory(name="v2")})
    assert r.status_code == 200
    assert r.json()["revision"] == 2
    assert r.json()["name"] == "v2"
    # old revision still fetchable
    old = c.get(f"/api/scenarios/{e['id']}", params={"revision": 1}).json()
    assert old["name"] == "case"
    latest = c.get(f"/api/scenarios/{e['id']}").json()
    assert latest["revision"] == 2


def test_stale_revision_rejected_with_409(client):
    c, _ = client
    e = create(client)
    sid = e["id"]
    # client A advances 1 -> 2
    r1 = c.put(f"/api/scenarios/{sid}",
               json={"expected_revision": 1, "scenario": body_factory(name="A")})
    assert r1.status_code == 200
    # client B still based on 1 -> must be rejected, carries current rev
    r2 = c.put(f"/api/scenarios/{sid}",
               json={"expected_revision": 1, "scenario": body_factory(name="B")})
    assert r2.status_code == 409
    detail = r2.json()
    assert detail["current_revision"] == 2
    # B's payload did NOT overwrite A
    assert c.get(f"/api/scenarios/{sid}").json()["name"] == "A"


def test_bad_expected_revision(client):
    c, _ = client
    e = create(client)
    r = c.put(f"/api/scenarios/{e['id']}",
              json={"expected_revision": 0, "scenario": body_factory()})
    assert r.status_code in (409, 422)
    r = c.put(f"/api/scenarios/{e['id']}",
              json={"expected_revision": 99, "scenario": body_factory()})
    assert r.status_code == 409


def test_save_nonexistent_404(client):
    c, _ = client
    r = c.put("/api/scenarios/nope",
              json={"expected_revision": 1, "scenario": body_factory()})
    assert r.status_code == 404


# --------------------------------------------------------------------------
# concurrent saves: exactly one of many racing writers may win
# ------------------------------------------------------------------

def test_concurrent_saves_only_one_wins_per_revision(client):
    _, store = client
    e = create(client)
    sid = e["id"]
    payload = body_factory()
    results = []
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        try:
            store.save(sid, {**payload, "name": f"w{i}"}, expected_revision=1)
            results.append(("ok", i))
        except RevisionConflict as exc:
            results.append(("conflict", exc.current))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    oks = [r for r in results if r[0] == "ok"]
    conflicts = [r for r in results if r[0] == "conflict"]
    assert len(oks) == 1
    assert len(conflicts) == 7
    assert all(r[1] == 2 for r in conflicts)
    assert store.current_revision(sid) == 2


def test_serialized_chain_many_writers(client):
    _, store = client
    e = create(client)
    sid = e["id"]
    payload = body_factory()
    cur = {"rev": 1}
    lock = threading.Lock()
    wins = {"n": 0}

    def worker():
        # retry-loop: each stale writer refreshes expected rev and retries,
        # simulating a well-behaved client; all updates must land exactly once
        for _ in range(20):
            with lock:
                expected = cur["rev"]
            try:
                rev = store.save(sid, payload, expected_revision=expected)["revision"]
            except RevisionConflict:
                continue
            with lock:
                if cur["rev"] == expected:
                    cur["rev"] = rev
                    wins["n"] += 1
            return

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cur["rev"] == 11
    assert wins["n"] == 10


# --------------------------------------------------------------------------
# solve: evidence + stale semantics
# ------------------------------------------------------------------

def test_solve_response_evidence(client):
    c, _ = client
    e = create(client)
    r = c.post(f"/api/scenarios/{e['id']}/solve")
    assert r.status_code == 200
    data = r.json()
    assert data["revision"] == 1
    sol = data["solution"]
    assert sol["feasible"]
    assert len(sol["periods"]) == 8
    # per-period physical evidence
    soc = e["initial_soc"]
    for i, pr in enumerate(sol["periods"]):
        assert pr["period"] == i
        assert pr["soc_start"] == soc
        assert not (pr["charge"] > 0 and pr["discharge"] > 0)
        net = e["load"][i] - e["pv"][i]
        assert pr["discharge"] <= max(0, net)
        assert pr["soc_end"] == soc + pr["charge"] - pr["discharge"]
        residual = net - pr["discharge"] + pr["charge"]
        assert pr["purchase"] == max(0, residual)
        assert pr["curtail"] == max(0, -residual)
        assert pr["cost"] == pr["purchase"] * e["price"][i]
        assert pr["load"] == e["load"][i]
        assert pr["pv"] == e["pv"][i]
        soc = pr["soc_end"]
    assert soc >= e["terminal_min_soc"]
    assert sol["final_soc"] == soc
    assert sol["total_cost"] == sum(p["cost"] for p in sol["periods"])
    assert sol["total_purchase"] == sum(p["purchase"] for p in sol["periods"])


def test_solve_specific_revision_is_stable(client):
    """Solving revision 1 after revision 2 exists must still echo revision 1
    and its plan (so an old plan can never silently describe the new rev)."""
    c, _ = client
    e = create(client)
    sid = e["id"]
    c.put(f"/api/scenarios/{sid}",
          json={"expected_revision": 1, "scenario": body_factory(name="v2")})
    r1 = c.post(f"/api/scenarios/{sid}/solve", params={"revision": 1}).json()
    r2 = c.post(f"/api/scenarios/{sid}/solve").json()
    assert r1["revision"] == 1
    assert r2["revision"] == 2
    # different inputs -> plans may differ; revisions never swapped
    assert r1["revision"] != r2["revision"]


def test_solve_infeasible(client):
    c, _ = client
    # 8 periods, max 1 unit/period charge from empty -> cannot reach 9
    e = create(client, capacity=10, max_charge=1, initial_soc=0, terminal_min_soc=9,
               load=[3] * 8, pv=[0] * 8)
    r = c.post(f"/api/scenarios/{e['id']}/solve").json()
    assert r["solution"]["feasible"] is False
    assert r["solution"]["reason"]
    assert r["solution"]["periods"] == []


def test_adhoc_solve_does_not_persist(client):
    c, store = client
    before = len(store.list_scenarios())
    r = c.post("/api/solve", json=body_factory())
    assert r.status_code == 200
    assert r.json()["revision"] is None
    assert len(store.list_scenarios()) == before


def test_list_and_delete(client):
    c, _ = client
    e = create(client)
    lst = c.get("/api/scenarios").json()
    assert any(s["id"] == e["id"] and s["revision"] == 1 and s["periods"] == 8 for s in lst)
    assert c.delete(f"/api/scenarios/{e['id']}").status_code == 204
    assert c.get(f"/api/scenarios/{e['id']}").status_code == 404


def test_index_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "虚拟电池" in r.text
    assert c.get("/static/style.css").status_code == 200
    assert c.get("/static/vendor/vue.global.prod.js").status_code == 200
