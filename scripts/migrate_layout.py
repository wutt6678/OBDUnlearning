"""One-shot (idempotent) layout migration for scripts/ and outputs/.

Target layout:
  scripts/baselines/  -- run_*_baselines.sh, run_all_baselines.sh,
                         materialize_baselines_all_models.py
  scripts/sweeps/     -- all other run_*.sh + materialize_*.py
  scripts/plots/      -- all plot_*.py
  scripts/            -- core utilities stay at the root

  outputs/sweeps/<model>/<sweep_suffix>/...   (was outputs/<model>_<suffix>/...)
  outputs/sweeps/<model>/llm_model_unlearning_method_default_sweep/...
  outputs/baselines/, outputs/finetune/, outputs/old/  -- unchanged

Steps (all safe to re-run):
  1. Move scripts into category subdirs (git mv, fallback to plain move).
  2. Patch moved .sh: ROOT_DIR one level deeper, cross-script references.
  3. Patch moved .py: parents[1] -> parents[2].
  4. Rewrite outputs/<old> paths to the new layout in configs/, scripts/,
     finetune/ (excluded roots make finetune/baselines refs no-ops).
  5. Move existing output directories on disk accordingly.
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"

CORE_SCRIPTS = {
    "run_open_tofu.py",
    "evaluate_benchmark.py",
    "finetune_tofu.py",
    "download_assets.py",
    "check_model_compat_shims.py",
    "generate_muse_examples.py",
    "scan_diff_thresholds.py",
    "migrate_layout.py",
}

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
MODELS_BY_LEN = sorted(MODELS, key=len, reverse=True)

EXCLUDED_OUTPUT_ROOTS = {"finetune", "baselines", "old", "sweeps"}
DEFAULT_SWEEP_DIR = "llm_model_unlearning_method_default_sweep"

# Matches outputs/<path> literals in yaml/sh/py text.
OUTPUT_PATH_RE = re.compile(r"(?<![\w./])outputs/([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*)")
SH_CROSSREF_RE = re.compile(r"scripts/(run_[a-z0-9_${}]+\.sh)")
SH_MATERIALIZER_RE = re.compile(r"scripts/(materialize_[a-z0-9_]+\.py)")
OLD_ROOT_DIR = '$(dirname "${BASH_SOURCE[0]}")/..'
NEW_ROOT_DIR = '$(dirname "${BASH_SOURCE[0]}")/../..'


def map_output_path(rel: str) -> str | None:
    """Map an outputs-relative path to the new layout, or None if untouched."""
    parts = rel.split("/")
    top = parts[0]
    if top in EXCLUDED_OUTPUT_ROOTS:
        return None
    if top == DEFAULT_SWEEP_DIR:
        if len(parts) >= 2 and parts[1].endswith("_sweep"):
            model = parts[1][: -len("_sweep")]
            return "/".join(["outputs", "sweeps", model, top] + parts[2:])
        return None
    for model in MODELS_BY_LEN:
        if top.startswith(model + "_"):
            suffix = top[len(model) + 1:]
            return "/".join(["outputs", "sweeps", model, suffix] + parts[1:])
    return None


def rewrite_outputs_in_text(text: str) -> tuple[str, int]:
    count = 0

    def sub(match: re.Match) -> str:
        nonlocal count
        new = map_output_path(match.group(1))
        if new is None:
            return match.group(0)
        count += 1
        return new

    return OUTPUT_PATH_RE.sub(sub, text), count


def classify_script(name: str) -> str | None:
    """Return target subdir for a scripts/ root file, or None to keep in place."""
    if name in CORE_SCRIPTS or name.startswith("__"):
        return None
    if (
        name == "materialize_baselines_all_models.py"
        or name == "run_all_baselines.sh"
        or (name.startswith("run_") and name.endswith("_baselines.sh"))
    ):
        return "baselines"
    if name.startswith("plot_") and name.endswith(".py"):
        return "plots"
    if (name.startswith("run_") and name.endswith(".sh")) or (
        name.startswith("materialize_") and name.endswith(".py")
    ):
        return "sweeps"
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
    print("[1/5] Moving scripts into category subdirs")
    moved: list[Path] = []
    for path in sorted(SCRIPTS.iterdir()):
        if not path.is_file():
            continue
        subdir = classify_script(path.name)
        if subdir is None:
            continue
        dst = SCRIPTS / subdir / path.name
        move_file(path, dst)
        moved.append(dst)
    print(f"  moved {len(moved)} files")
    return moved


def script_subdir_for(name: str) -> str:
    if name.endswith("_baselines.sh"):
        return "baselines"
    if name.startswith("materialize_"):
        return "baselines" if name == "materialize_baselines_all_models.py" else "sweeps"
    return "sweeps"


def step2_patch_shell(moved: list[Path]) -> None:
    print("[2/5] Patching moved .sh files")
    for path in (p for p in moved if p.suffix == ".sh"):
        text = path.read_text(encoding="utf-8")
        original = text
        text = text.replace(OLD_ROOT_DIR, NEW_ROOT_DIR)
        text = SH_CROSSREF_RE.sub(
            lambda m: f"scripts/{script_subdir_for(m.group(1))}/{m.group(1)}", text
        )
        text = SH_MATERIALIZER_RE.sub(
            lambda m: f"scripts/{script_subdir_for(m.group(1))}/{m.group(1)}", text
        )
        if text != original:
            path.write_text(text, encoding="utf-8")


def step3_patch_python(moved: list[Path]) -> None:
    print("[3/5] Patching moved .py files")
    for path in (p for p in moved if p.suffix == ".py"):
        text = path.read_text(encoding="utf-8")
        new = text.replace(
            "Path(__file__).resolve().parents[1]",
            "Path(__file__).resolve().parents[2]",
        )
        if new != text:
            path.write_text(new, encoding="utf-8")


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
    print("[4/5] Rewriting outputs/ paths in configs, scripts, finetune")
    total = 0
    changed_files = 0
    unmapped: set[str] = set()
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        new, count = rewrite_outputs_in_text(text)
        for match in OUTPUT_PATH_RE.finditer(text):
            if map_output_path(match.group(1)) is None:
                top = match.group(1).split("/")[0]
                if top not in EXCLUDED_OUTPUT_ROOTS:
                    unmapped.add(top)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed_files += 1
            total += count
    print(f"  rewrote {total} path references in {changed_files} files")
    if unmapped:
        print(f"  WARNING: unmapped outputs/ roots left as-is: {sorted(unmapped)}")


def move_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  skip (exists): {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  moved {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def step5_move_outputs() -> None:
    print("[5/5] Moving existing output directories")
    if not OUTPUTS.is_dir():
        return
    for child in sorted(OUTPUTS.iterdir()):
        if not child.is_dir() or child.name in EXCLUDED_OUTPUT_ROOTS:
            continue
        if child.name == DEFAULT_SWEEP_DIR:
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and sub.name.endswith("_sweep"):
                    model = sub.name[: -len("_sweep")]
                    move_dir(sub, OUTPUTS / "sweeps" / model / DEFAULT_SWEEP_DIR)
            if not any(child.iterdir()):
                child.rmdir()
            continue
        rel = f"outputs/{child.name}"
        new = map_output_path(child.name)
        if new is None:
            print(f"  WARNING: cannot map {rel}; leaving in place")
            continue
        move_dir(child, ROOT / new)


def main() -> None:
    moved = step1_move_scripts()
    step2_patch_shell(moved)
    step3_patch_python(moved)
    step4_rewrite_output_paths()
    step5_move_outputs()
    print("Done.")


if __name__ == "__main__":
    main()
