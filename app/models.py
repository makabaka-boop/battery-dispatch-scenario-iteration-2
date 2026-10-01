"""Pydantic schemas for scenario API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


MIN_PERIODS = 8
MAX_PERIODS = 48
MAX_CAPACITY = 50


class ScenarioInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    load: List[int]
    pv: List[int]
    price: List[int]
    capacity: int = Field(ge=1, le=MAX_CAPACITY)
    initial_soc: int = Field(ge=0)
    terminal_min_soc: int = Field(ge=0)
    max_charge: int = Field(ge=1)
    # Optional hard cap on the sum of purchase over all periods.  Old stored
    # revisions simply lack the key; it then reads back as None and solving
    # follows the unconstrained optimum.
    purchase_budget: Optional[int] = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        return v

    @field_validator("load", "pv", "price")
    @classmethod
    def _nonneg_series(cls, v: List[int]) -> List[int]:
        if any(x < 0 for x in v):
            raise ValueError("series values must be non-negative integers")
        return v

    @model_validator(mode="after")
    def _check(self) -> "ScenarioInput":
        n = len(self.load)
        if not (MIN_PERIODS <= n <= MAX_PERIODS):
            raise ValueError(
                f"expected between {MIN_PERIODS} and {MAX_PERIODS} periods, got {n}"
            )
        if len(self.pv) != n or len(self.price) != n:
            raise ValueError("load, pv and price must have the same length")
        if self.initial_soc > self.capacity:
            raise ValueError("initial_soc must not exceed capacity")
        if self.terminal_min_soc > self.capacity:
            raise ValueError("terminal_min_soc must not exceed capacity")
        if self.max_charge > self.capacity:
            raise ValueError("max_charge must not exceed capacity")
        return self


class ScenarioRevision(ScenarioInput):
    id: str
    revision: int
    created_at: str
    updated_at: str


class PeriodOut(BaseModel):
    period: int
    load: int
    pv: int
    price: int
    soc_start: int
    charge: int
    discharge: int
    soc_end: int
    net_load: int
    purchase: int
    curtail: int
    cost: int
    # Purchase budget still unspent after this period (None when the scenario
    # has no budget): remaining[t] == budget - sum(purchase[0..t]), so it
    # cross-checks against cumulative purchase.
    remaining_budget: Optional[int] = None


class SolutionOut(BaseModel):
    feasible: bool
    total_cost: int
    total_purchase: int
    final_soc: Optional[int] = None
    reason: Optional[str] = None
    periods: List[PeriodOut]
    # Echoed scenario cap (None when uncapped) and, when the physical problem
    # is feasible but the cap is too small, the minimum total purchase that
    # satisfying the physical constraints would require.
    purchase_budget: Optional[int] = None
    required_min_purchase: Optional[int] = None


class SolveResponse(BaseModel):
    scenario_id: str
    revision: int
    solution: SolutionOut


class ScenarioSummary(BaseModel):
    id: str
    revision: int
    name: str
    periods: int
    updated_at: str


class ConflictOut(BaseModel):
    detail: str
    current_revision: int
