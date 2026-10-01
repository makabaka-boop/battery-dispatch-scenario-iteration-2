"""FastAPI service for the virtual-battery scenario solver.

API:
  POST   /api/scenarios                 create (revision starts at 1)
  GET    /api/scenarios                 list latest revisions
  GET    /api/scenarios/{sid}           get a revision (?revision=N, default latest)
  PUT    /api/scenarios/{sid}           save edit, body carries expected_revision
  DELETE /api/scenarios/{sid}
  POST   /api/scenarios/{sid}/solve     solve a stored revision (?revision=N)
  POST   /api/solve                     solve an ad-hoc scenario without saving
  GET    /                              Vue scenario page

The service never talks to real power equipment; all data is simulated.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import ScenarioInput, SolutionOut
from .solver import PeriodResult, Solution, normalize, solve
from .storage import RevisionConflict, ScenarioNotFound, ScenarioStore

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.normpath(os.path.join(HERE, "..", "web"))
DB_PATH = os.environ.get("VBAT_DB", os.path.join(HERE, "..", "data", "scenarios.json"))


class SaveBody(BaseModel):
    expected_revision: int = Field(ge=1)
    scenario: ScenarioInput


def _solution_out(sol: Solution, entry: Optional[dict] = None) -> SolutionOut:
    out = SolutionOut(
        feasible=sol.feasible,
        total_cost=sol.total_cost,
        total_purchase=sol.total_purchase,
        final_soc=sol.final_soc,
        reason=sol.reason,
        periods=[
            {
                "period": r.period,
                "load": entry["load"][r.period],
                "pv": entry["pv"][r.period],
                "price": entry["price"][r.period],
                "soc_start": r.soc_start,
                "charge": r.charge,
                "discharge": r.discharge,
                "soc_end": r.soc_end,
                "net_load": r.net_load,
                "purchase": r.purchase,
                "curtail": r.curtail,
                "cost": r.cost,
            }
            for r in sol.periods
        ],
    )
    return out


def _solve_entry(entry: dict) -> Solution:
    p = normalize(
        load=entry["load"],
        pv=entry["pv"],
        price=entry["price"],
        capacity=entry["capacity"],
        initial_soc=entry["initial_soc"],
        terminal_min_soc=entry["terminal_min_soc"],
        max_charge=entry["max_charge"],
    )
    return solve(p)


def create_app(store: Optional[ScenarioStore] = None) -> FastAPI:
    app = FastAPI(title="Virtual Battery Solver", version="1.0.0")
    app.state.store = store or ScenarioStore(DB_PATH)

    @app.exception_handler(RevisionConflict)
    async def _conflict_handler(request, exc: RevisionConflict):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "current_revision": exc.current,
            },
        )

    @app.exception_handler(ScenarioNotFound)
    async def _not_found_handler(request, exc: ScenarioNotFound):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": "scenario not found"})

    @app.post("/api/scenarios", status_code=201)
    def create_scenario(scenario: ScenarioInput):
        entry = app.state.store.create(scenario.model_dump())
        return entry

    @app.get("/api/scenarios")
    def list_scenarios():
        return app.state.store.list_scenarios()

    @app.get("/api/scenarios/{sid}")
    def get_scenario(sid: str, revision: Optional[int] = Query(default=None, ge=1)):
        return app.state.store.get(sid, revision)

    @app.put("/api/scenarios/{sid}")
    def save_scenario(sid: str, body: SaveBody):
        try:
            return app.state.store.save(
                sid, body.scenario.model_dump(), body.expected_revision
            )
        except RevisionConflict:
            raise

    @app.delete("/api/scenarios/{sid}", status_code=204)
    def delete_scenario(sid: str):
        app.state.store.delete(sid)

    @app.post("/api/scenarios/{sid}/solve")
    def solve_stored(sid: str, revision: Optional[int] = Query(default=None, ge=1)):
        entry = app.state.store.get(sid, revision)
        sol = _solve_entry(entry)
        return {
            "scenario_id": sid,
            "revision": entry["revision"],
            "solution": _solution_out(sol, entry).model_dump(),
        }

    @app.post("/api/solve")
    def solve_adhoc(scenario: ScenarioInput):
        payload = scenario.model_dump()
        sol = _solve_entry(payload)
        return {"scenario_id": None, "revision": None,
                "solution": _solution_out(sol, payload).model_dump()}

    # --- static frontend (last, so /api routes take precedence) -----------
    if os.path.isdir(WEB_DIR):
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(os.path.join(WEB_DIR, "index.html"))

    return app


app = create_app()
