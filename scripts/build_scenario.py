"""Build ViZDoom scenario WAD variants from ACS sources.

CLI:
    uv run python scripts/build_scenario.py <scenario_dir> <variant> [--force]

Library:
    from scripts.build_scenario import ensure_built
    cfg_path = ensure_built("remedy_rush_coop", "shared_reward_v1")

Input layout (scenarios/):
    scenarios/<scenario_dir>/base.ref          # one line: base scenario name
    scenarios/<scenario_dir>/base.cfg          # optional; local pristine copy
    scenarios/<scenario_dir>/base.acs          # optional; for diffing
    scenarios/<scenario_dir>/base.md           # optional; docs
    scenarios/<scenario_dir>/variants/<variant>/
        scripts.acs                            # required; compiled into BEHAVIOR
        decorate.txt                           # optional; injected as top-level DECORATE lump
        scenario.cfg                           # optional; full cfg override
        README.md                              # optional; variant docs

CFG resolution order (first match wins):
    1. scenarios/<scenario_dir>/variants/<variant>/scenario.cfg
    2. scenarios/<scenario_dir>/base.cfg
    3. <vizdoom_install>/scenarios/<base_scenario>.cfg

Output layout (scenarios_built/, gitignored):
    scenarios_built/<scenario_dir>/<variant>/<base_scenario>.wad
    scenarios_built/<scenario_dir>/<variant>/<base_scenario>.cfg
    scenarios_built/<scenario_dir>/<variant>/build.json  # hashes for cache-invalidation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
BUILT_DIR = PROJECT_ROOT / "scenarios_built"

VARIANT_ACS_NAME = "scripts.acs"
VARIANT_CFG_NAME = "scenario.cfg"
VARIANT_DECORATE_NAME = "decorate.txt"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class BuildInputs:
    scenario_dir: str
    variant: str
    base_scenario: str
    base_wad: Path
    cfg_source: Path      # variant-local, scenario-local, or upstream
    cfg_origin: str       # "variant" | "scenario" | "upstream"
    acs_source: Path
    decorate_source: Path | None  # optional DECORATE lump source
    acc_bin: Path


def resolve_base_scenario(scenario_dir: str) -> str:
    """Public: read scenarios/<scenario_dir>/base.ref and return the base name."""
    return _resolve_base_scenario(scenario_dir)


def _resolve_base_scenario(scenario_dir: str) -> str:
    ref_path = SCENARIOS_DIR / scenario_dir / "base.ref"
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Missing {ref_path.relative_to(PROJECT_ROOT)}. "
            "Create it with a single line naming the base ViZDoom scenario "
            "(e.g. 'remedy_rush_coop_multi_agent')."
        )
    name = ref_path.read_text(encoding="utf-8").strip()
    if not name:
        raise ValueError(f"{ref_path} is empty; expected a base scenario name.")
    if "\n" in name or "/" in name or name.endswith(".wad") or name.endswith(".cfg"):
        raise ValueError(
            f"{ref_path} must contain exactly one scenario name without extensions "
            f"or paths; got: {name!r}"
        )
    return name


def _resolve_base_wad(base_scenario: str) -> Path:
    import vizdoom as vzd

    wad = Path(vzd.scenarios_path) / f"{base_scenario}.wad"
    if not wad.is_file():
        raise FileNotFoundError(f"Base WAD not found: {wad}")
    return wad


def _resolve_cfg(scenario_dir: str, variant: str, base_scenario: str) -> tuple[Path, str]:
    """Return (cfg_path, origin) with origin ∈ {'variant', 'scenario', 'upstream'}."""
    variant_cfg = SCENARIOS_DIR / scenario_dir / "variants" / variant / VARIANT_CFG_NAME
    if variant_cfg.is_file():
        return variant_cfg, "variant"

    scenario_cfg = SCENARIOS_DIR / scenario_dir / "base.cfg"
    if scenario_cfg.is_file():
        return scenario_cfg, "scenario"

    import vizdoom as vzd

    upstream_cfg = Path(vzd.scenarios_path) / f"{base_scenario}.cfg"
    if upstream_cfg.is_file():
        return upstream_cfg, "upstream"

    raise FileNotFoundError(
        f"No cfg found for {scenario_dir}/{variant}. Looked in:\n"
        f"  1. {variant_cfg.relative_to(PROJECT_ROOT)}\n"
        f"  2. {scenario_cfg.relative_to(PROJECT_ROOT)}\n"
        f"  3. {upstream_cfg}"
    )


def _gather_inputs(scenario_dir: str, variant: str) -> BuildInputs:
    from vizdoom.scenario_building.acs_patch import locate_acc

    base_scenario = _resolve_base_scenario(scenario_dir)
    base_wad = _resolve_base_wad(base_scenario)

    variant_dir = SCENARIOS_DIR / scenario_dir / "variants" / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(
            f"Variant directory not found: {variant_dir.relative_to(PROJECT_ROOT)}"
        )
    acs_source = variant_dir / VARIANT_ACS_NAME
    if not acs_source.is_file():
        raise FileNotFoundError(
            f"Missing {acs_source.relative_to(PROJECT_ROOT)} — each variant "
            f"directory must contain a '{VARIANT_ACS_NAME}' file."
        )

    cfg_source, cfg_origin = _resolve_cfg(scenario_dir, variant, base_scenario)
    acc_bin, _ = locate_acc()

    decorate_source = variant_dir / VARIANT_DECORATE_NAME
    decorate_source = decorate_source if decorate_source.is_file() else None

    return BuildInputs(
        scenario_dir=scenario_dir,
        variant=variant,
        base_scenario=base_scenario,
        base_wad=base_wad,
        cfg_source=cfg_source,
        cfg_origin=cfg_origin,
        acs_source=acs_source,
        decorate_source=decorate_source,
        acc_bin=acc_bin,
    )


def _manifest(inputs: BuildInputs, built_wad_name: str, built_cfg_name: str) -> dict:
    hashes = {
        "acs_source": _sha256_file(inputs.acs_source),
        "base_wad": _sha256_file(inputs.base_wad),
        "cfg_source": _sha256_file(inputs.cfg_source),
        "acc_bin": _sha256_file(inputs.acc_bin),
    }
    if inputs.decorate_source is not None:
        hashes["decorate_source"] = _sha256_file(inputs.decorate_source)
    return {
        "scenario_dir": inputs.scenario_dir,
        "variant": inputs.variant,
        "base_scenario": inputs.base_scenario,
        "built_wad_name": built_wad_name,
        "built_cfg_name": built_cfg_name,
        "cfg_origin": inputs.cfg_origin,
        "hashes": hashes,
    }


def _is_cached(out_dir: Path, manifest: dict) -> bool:
    manifest_path = out_dir / "build.json"
    if not manifest_path.is_file():
        return False
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if existing.get("hashes") != manifest["hashes"]:
        return False
    if existing.get("cfg_origin") != manifest["cfg_origin"]:
        return False
    wad = out_dir / existing.get("built_wad_name", "")
    cfg = out_dir / existing.get("built_cfg_name", "")
    return wad.is_file() and cfg.is_file()


def _inject_decorate(wad_path: Path, decorate_bytes: bytes) -> None:
    """Add or replace the top-level DECORATE lump in an existing WAD."""
    from omg import WAD, Lump

    wad = WAD(str(wad_path))
    wad.data["DECORATE"] = Lump(decorate_bytes)
    wad.to_file(str(wad_path))


def _rewrite_cfg(base_cfg: Path, out_cfg: Path, built_wad: Path) -> None:
    """Copy base cfg, rewriting doom_scenario_path to the bare WAD filename.

    ViZDoom resolves doom_scenario_path relative to the cfg's directory and
    prepends that directory even when the value is already absolute, so we
    emit just the basename (the built WAD is always colocated with the cfg).
    """
    replacement = f"doom_scenario_path = {built_wad.name}\n"
    out_lines: list[str] = []
    replaced = False
    for line in base_cfg.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.lstrip()
        if not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip().replace("_", "").lower()
            if key == "doomscenariopath":
                out_lines.append(replacement)
                replaced = True
                continue
        out_lines.append(line)
    if not replaced:
        out_lines.append(replacement)
    out_cfg.write_text("".join(out_lines), encoding="utf-8")


def ensure_built(
    scenario_dir: str,
    variant: str,
    *,
    force: bool = False,
) -> Path:
    """Build (or reuse a cached build of) a scenario variant.

    Returns the absolute path to the generated .cfg, suitable for
    DoomGame.load_config or vizdoom.pettingzoo_wrapper.make(config_file=...).
    """
    from vizdoom.scenario_building.acs_patch import build_patched_wad

    inputs = _gather_inputs(scenario_dir, variant)
    out_dir = BUILT_DIR / scenario_dir / variant
    built_wad_name = f"{inputs.base_scenario}.wad"
    built_cfg_name = f"{inputs.base_scenario}.cfg"
    built_wad = out_dir / built_wad_name
    built_cfg = out_dir / built_cfg_name

    manifest = _manifest(inputs, built_wad_name, built_cfg_name)

    if not force and _is_cached(out_dir, manifest):
        return built_cfg.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    acs_text = inputs.acs_source.read_text(encoding="utf-8")
    build_patched_wad(inputs.base_wad, acs_text, built_wad)
    if inputs.decorate_source is not None:
        _inject_decorate(built_wad, inputs.decorate_source.read_bytes())
    _rewrite_cfg(inputs.cfg_source, built_cfg, built_wad)
    (out_dir / "build.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return built_cfg.resolve()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "scenario_dir",
        help="Directory under scenarios/ (e.g. remedy_rush_coop).",
    )
    ap.add_argument(
        "variant",
        help=f"Variant directory name (scenarios/<scenario_dir>/variants/<variant>/{VARIANT_ACS_NAME}).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the cache is fresh.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = ensure_built(args.scenario_dir, args.variant, force=args.force)
    print(f"[build_scenario] {cfg_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
