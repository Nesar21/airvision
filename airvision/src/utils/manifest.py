from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import RUN_MANIFEST_PATH, EXPERIMENT_FAMILY


@dataclass
class RunEntry:
    run_id: str
    family: str
    phase: str                    # e.g. "phase0_env", "phase1_preproc", ...
    stage: str                    # e.g. "haze_pretrain", "ind_nep_fold0"
    description: str
    timestamp: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _ensure_manifest_file(path: Path) -> None:
    if not path.exists():
        path.write_text(json.dumps({"runs": []}, indent=2))


def load_manifest(path: Path = RUN_MANIFEST_PATH) -> Dict[str, Any]:
    _ensure_manifest_file(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any], path: Path = RUN_MANIFEST_PATH) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def log_run(
    phase: str,
    stage: str,
    description: str,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    notes: str = "",
    path: Path = RUN_MANIFEST_PATH,
) -> str:
    """
    Append a run entry to run_manifest.json and return run_id.
    """
    config = config or {}
    metrics = metrics or {}

    run_id = str(uuid.uuid4())

    entry = RunEntry(
        run_id=run_id,
        family=EXPERIMENT_FAMILY,
        phase=phase,
        stage=stage,
        description=description,
        config=config,
        metrics=metrics,
        notes=notes,
    )

    manifest = load_manifest(path)
    manifest.setdefault("runs", []).append(asdict(entry))
    save_manifest(manifest, path)

    print(f"[MANIFEST] Logged run_id={run_id} phase={phase} stage={stage}")
    return run_id


if __name__ == "__main__":
    # simple self-test for Phase 0
    rid = log_run(
        phase="phase0_env",
        stage="sanity_check",
        description="Test entry from manifest self-test.",
        config={"python": ">=3.11"},
        metrics={"ok": True},
        notes="If you see this, manifest is working.",
    )
    print("Test run id:", rid)
