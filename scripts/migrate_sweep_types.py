"""One-shot (idempotent) migration: subdivide sweeps by type.

Target layout (applies to scripts/sweeps/, scripts/plots/, outputs/sweeps/):
  alpha/   -- stem contains _alpha_
  beta/    -- stem contains _beta_
  main/    -- stem contains _main_ or main_sweep
  masks/   -- mask_construction_*, dual_mask_ablation_*, *dynamic_k5*
  default/ -- *llm_model_unlearning_method_default_sweep*
  root     -- umbrella dispatchers/materializers/plots and unmatched files

  outputs/sweeps/<model>/<type>/<suffix>/...  (was outputs/sweeps/<model>/<suffix>/...)
  The reserved aggregate dir outputs/sweeps/llm_model_unlearning_method_default_sweep/
  (plots/summary for the default sweep) stays untouched.

Steps (all safe to re-run):
  1. Move scripts/sweeps/ and scripts/plots/ files into type subdirs
     (git mv, fallback to plain move).
  2. Patch moved .sh: ROOT_DIR one level deeper, cross-script references.
  3. Patch moved .py: parents[2] -> parents[3].
  4. Insert the type segment into outputs/sweeps/<model>/<suffix> literals
     in configs/, scripts/, finetune/.
  5. Patch the two files that build default-sweep paths at runtime.
  6. Move existing output directories on disk accordingly.
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SWEEPS = SCRIPTS / "sweeps"
PLOTS = SCRIPTS / "plots"
OUTPUTS = ROOT / "outputs"
SWEEP_OUTPUTS = OUTPUTS / "sweeps"

# 21 baseline models plus legacy qwen2.5 / qwen3-4b sweep models.
MODELS = [
    "deepseek_r1_distill_llama_8b",
    "deepseek_r1_distill_qwen_1_5b",
    "deepseek_r1_distill_qwen_7b",
    "gemma4_e2b",
    "llama3_1_8b",
    "llama3_1b",
    "llama3_2_3b",
    "llama3_3b",
    "llama3_8b",
    "ministral3_3b_instruct",
    "mistral7b_instruct_v03",
    "phi4_mini_instruct",
    "qwen2_5_0_5b_instruct",
    "qwen2_5_1_5b_instruct",
    "qwen2_5_3b_instruct",
    "qwen2_5_7b_instruct",
    "qwen3_4b_instruct_2507",
    "qwen3_5_0_8b",
    "qwen3_5_27b",
    "qwen3_5_2b",
    "qwen3_5_4b",
    "qwen3_5_9b",
    "qwen3_6_27b",
    "smollm3_3b",
    "zai_chatglm3_6b",
    "zai_glm_4_9b",
]

TYPE_BUCKETS = {"alpha", "beta", "main", "masks", "default", "misc"}
DEFAULT_SWEEP_TOKEN = "llm_model_unlearning_method_default_sweep"

# Inserts the type segment into outputs/sweeps/<model>/<suffix> literals.
OUTPUT_PATH_RE = re.compile(r"(?<![\w./])outputs/sweeps/([a-z0-9_]+)/([a-z0-9_]+)")
SH_CROSSREF_RE = re.compile(r"scripts/sweeps/(run_[a-z0-9_${}]+\.sh)")
SH_MATERIALIZER_RE = re.compile(r"scripts/sweeps/(materialize_[a-z0-9_]+\.py)")
OLD_ROOT_DIR = '$(dirname "${BASH_SOURCE[0]}")/../..'
NEW_ROOT_DIR = '$(dirname "${BASH_SOURCE[0]}")/../../..'


def sweep_type(name: str) -> str | None:
    """Return the type bucket for a script name / output suffix, or None."""
    stem = re.sub(r"\.(sh|py)$", "", name)
    stem = re.sub(r"^(run|materialize|plot)_", "", stem)
    if DEFAULT_SWEEP_TOKEN in stem:
        return "default"
    if "mask_construction" in stem or "dual_mask_ablation" in stem or "dynamic_k5" in stem:
        return "masks"
    if "_alpha_" in stem:
        return "alpha"
    if "_beta_" in stem:
        return "beta"
    if "_main_" in stem or "main_sweep" in stem:
        return "main"
    return None


def move_file(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  skip (exists): {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "mv", str(src), str(dst)], cwd=ROOT, capture_output=True
    )
    if result.returncode != 0:
        shutil.move(str(src), str(dst))


def step1_move_scripts() -> list[Path]:
    print("[1/6] Moving sweep/plot scripts into type subdirs")
    moved: list[Path] = []
    for base in (SWEEPS, PLOTS):
        for path in sorted(base.iterdir()):
            if not path.is_file():
                continue
            bucket = sweep_type(path.name)
            if bucket is None:
                continue
            dst = base / bucket / path.name
            move_file(path, dst)
            moved.append(dst)
    print(f"  moved {len(moved)} files")
    return moved


def step2_patch_shell(moved: list[Path]) -> None:
    print("[2/6] Patching moved .sh files")
    changed = 0
    for path in (p for p in moved if p.suffix == ".sh"):
        text = path.read_text(encoding="utf-8")
        original = text
        text = text.replace(OLD_ROOT_DIR, NEW_ROOT_DIR)

        def sub_crossref(match: re.Match) -> str:
            name = match.group(1)
            bucket = sweep_type(name)
            if bucket is None:
                return match.group(0)
            return f"scripts/sweeps/{bucket}/{name}"

        text = SH_CROSSREF_RE.sub(sub_crossref, text)
        text = SH_MATERIALIZER_RE.sub(sub_crossref, text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1
    print(f"  patched {changed} files")


def step3_patch_python(moved: list[Path]) -> None:
    print("[3/6] Patching moved .py files")
    changed = 0
    for path in (p for p in moved if p.suffix == ".py"):
        text = path.read_text(encoding="utf-8")
        new = text.replace(
            "Path(__file__).resolve().parents[2]",
            "Path(__file__).resolve().parents[3]",
        )
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"  patched {changed} files")


def iter_text_files() -> list[Path]:
    files = list((ROOT / "configs").rglob("*.yaml"))
    files += [
        p
        for p in SCRIPTS.rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    files += list(SCRIPTS.rglob("*.sh"))
    finetune = ROOT / "finetune"
    if finetune.is_dir():
        files += [
            p
            for pattern in ("*.yaml", "*.py", "*.sh")
            for p in finetune.rglob(pattern)
            if "__pycache__" not in p.parts
        ]
    return files


def step4_rewrite_output_paths() -> None:
    print("[4/6] Inserting type segment into outputs/sweeps paths")
    total = 0
    changed_files = 0
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        count = 0

        def sub(match: re.Match) -> str:
            nonlocal count
            model, suffix = match.group(1), match.group(2)
            if model not in MODELS or suffix in TYPE_BUCKETS:
                return match.group(0)
            bucket = sweep_type(suffix)
            if bucket is None:
                return match.group(0)
            count += 1
            return f"outputs/sweeps/{model}/{bucket}/{suffix}"

        new = OUTPUT_PATH_RE.sub(sub, text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed_files += 1
            total += count
    print(f"  rewrote {total} path references in {changed_files} files")


def step5_patch_dynamic_default_paths() -> None:
    print("[5/6] Patching default-sweep scripts that build paths at runtime")
    materializer = SWEEPS / "default" / "materialize_llm_model_unlearning_method_default_sweep.py"
    plotter = PLOTS / "default" / "plot_llm_model_unlearning_method_default_sweep.py"
    if materializer.is_file():
        text = materializer.read_text(encoding="utf-8")
        new = text.replace(
            '"outputs/sweeps/{model}/llm_model_unlearning_method_default_sweep"',
            '"outputs/sweeps/{model}/default/llm_model_unlearning_method_default_sweep"',
        )
        if new != text:
            materializer.write_text(new, encoding="utf-8")
            print(f"  patched {materializer.relative_to(ROOT)}")
    else:
        print(f"  WARNING: missing {materializer.relative_to(ROOT)}")
    if plotter.is_file():
        text = plotter.read_text(encoding="utf-8")
        new = text.replace(
            "sweep_dir = model_dir / SWEEP_SUBDIR",
            'sweep_dir = model_dir / "default" / SWEEP_SUBDIR',
        )
        if new != text:
            plotter.write_text(new, encoding="utf-8")
            print(f"  patched {plotter.relative_to(ROOT)}")
    else:
        print(f"  WARNING: missing {plotter.relative_to(ROOT)}")


def step6_move_outputs() -> None:
    print("[6/6] Moving existing sweep output directories")
    if not SWEEP_OUTPUTS.is_dir():
        return
    for model_dir in sorted(SWEEP_OUTPUTS.iterdir()):
        if not model_dir.is_dir():
            continue
        for child in sorted(model_dir.iterdir()):
            if not child.is_dir() or child.name in TYPE_BUCKETS:
                continue
            bucket = sweep_type(child.name)
            if bucket is None:
                print(f"  WARNING: cannot classify {child.relative_to(ROOT)}; leaving in place")
                continue
            dst = model_dir / bucket / child.name
            if dst.exists():
                print(f"  skip (exists): {dst.relative_to(ROOT)}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(dst))
            print(f"  moved {child.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> None:
    moved = step1_move_scripts()
    step2_patch_shell(moved)
    step3_patch_python(moved)
    step4_rewrite_output_paths()
    step5_patch_dynamic_default_paths()
    step6_move_outputs()
    print("Done.")


if __name__ == "__main__":
    main()
