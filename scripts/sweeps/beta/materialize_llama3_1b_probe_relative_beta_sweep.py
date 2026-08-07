from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

runpy.run_path(
    str(ROOT / "scripts/materialize_llama3_1b_probe_relative_beta_objective_sweeps.py"),
    run_name="__main__",
)
