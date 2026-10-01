"""Thread-safe scenario storage with optimistic revision (revision) control.

A saved scenario keeps *all* revisions: every PUT appends a new revision.
Concurrent saves are serialized by a lock and each update must present the
revision it was based on; stale writes are rejected with RevisionConflict so
that one client can never silently overwrite another client's revision.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RevisionConflict(Exception):
    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(
            f"expected revision {expected}, current revision is {current}"
        )


class ScenarioNotFound(KeyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill fields introduced after older archives were written.

    ``purchase_budget`` did not exist in the first schema revision; reading
    such an archive yields ``None`` (unconstrained) rather than failing.
    """
    entry.setdefault("purchase_budget", None)
    return entry


def _entry(payload: Dict[str, Any], scenario_id: str, revision: int,
          created_at: Optional[str] = None) -> Dict[str, Any]:
    ts = _now()
    return {
        **payload,
        "id": scenario_id,
        "revision": revision,
        "created_at": created_at or ts,
        "updated_at": ts,
    }


class ScenarioStore:
    """In-memory store; optionally persisted to a JSON file under a lock."""

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._lock = threading.RLock()
        # id -> {"revisions": [entry, ...]}
        self._data: Dict[str, Dict[str, Any]] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    # -- persistence -------------------------------------------------------

    def _flush(self) -> None:
        if self._path:
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)

    # -- reads -------------------------------------------------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for sid, rec in self._data.items():
                latest = rec["revisions"][-1]
                out.append({
                    "id": sid,
                    "revision": latest["revision"],
                    "name": latest["name"],
                    "periods": len(latest["load"]),
                    "updated_at": latest["updated_at"],
                })
            return out

    def get(self, scenario_id: str, revision: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            rec = self._data.get(scenario_id)
            if rec is None:
                raise ScenarioNotFound(scenario_id)
            if revision is None:
                return _normalize_entry(dict(rec["revisions"][-1]))
            for rev in rec["revisions"]:
                if rev["revision"] == revision:
                    return _normalize_entry(dict(rev))
            raise ScenarioNotFound(f"{scenario_id}@r{revision}")

    def current_revision(self, scenario_id: str) -> int:
        return self.get(scenario_id)["revision"]

    # -- writes ------------------------------------------------------------

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scenario_id = uuid.uuid4().hex[:12]
        with self._lock:
            if scenario_id in self._data:  # astronomically unlikely
                raise RuntimeError("id collision")
            rev = _entry(payload, scenario_id, revision=1)
            self._data[scenario_id] = {"revisions": [rev]}
            self._flush()
            return dict(rev)

    def save(self, scenario_id: str, payload: Dict[str, Any],
             expected_revision: int) -> Dict[str, Any]:
        """Append a new revision only if the caller saw ``expected_revision``.

        Raises :class:`RevisionConflict` otherwise.
        """
        if not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        with self._lock:
            rec = self._data.get(scenario_id)
            if rec is None:
                raise ScenarioNotFound(scenario_id)
            current = rec["revisions"][-1]
            if current["revision"] != expected_revision:
                raise RevisionConflict(expected_revision, current["revision"])
            new_rev = _entry(
                payload,
                scenario_id,
                revision=current["revision"] + 1,
                created_at=current["created_at"],
            )
            rec["revisions"].append(new_rev)
            self._flush()
            return dict(new_rev)

    def delete(self, scenario_id: str) -> None:
        with self._lock:
            if scenario_id not in self._data:
                raise ScenarioNotFound(scenario_id)
            del self._data[scenario_id]
            self._flush()
