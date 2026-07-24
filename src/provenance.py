"""Experiment provenance + fail-fast schema enforcement (audit r4 #1, #12).

Stamping metadata is not protection unless readers ENFORCE it. Every result writer calls
`stamp()`; every analysis reader calls `require_schema()`. Analyses fail by default and must be
explicitly opted in to legacy / exploratory files.

Vocabulary
  experiment_type : transport | necessity | ablation_sweep | structure | align | layer_sweep | transport_sweep
  estimand        : what was actually intervened on -- the thing that must not be silently mixed
  analysis_status : validated       -> admissible as a primary paper claim
                    legacy_diagnostic -> old estimand, kept only as a diagnostic
                    exploratory     -> confounded / not-yet-rebuilt (sweeps)
"""
from __future__ import annotations

import subprocess

VALIDATED = "validated"
LEGACY = "legacy_diagnostic"
EXPLORATORY = "exploratory"

# Estimands, named so two different interventions can never share a heading.
E_DELTA = "matched_arithmetic_delta"            # h_B + QQ^T(h_en(a',b) - h_en(a,b))
E_ABSOLUTE = "absolute_carrier_reconstruction"  # legacy: replace subspace with a carrier reconstruction
E_ABLATION = "norm_matched_subspace_ablation"   # mean-ablate, controls energy-matched per case
E_LAYER_VULN = "heldout_layerwise_vulnerability"  # exploratory ablation-sweep peak
E_GEOMETRY = "representational_geometry"        # correlational fit/alignment (no intervention)


def git_metadata() -> dict:
    """Repo commit + worktree cleanliness. Two materially different implementations can share a
    schema_version, so the commit is what actually identifies the code that produced a result."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return {"code_commit": commit, "dirty_worktree": dirty}
    except Exception:
        return {"code_commit": None, "dirty_worktree": None}


def stamp(schema_version: str, experiment_type: str, estimand: str,
          analysis_status: str = VALIDATED, allow_dirty: bool = True, **extra) -> dict:
    """Build the provenance header every result file starts with. Set allow_dirty=False on
    production runs to refuse writing results from an uncommitted / unknown worktree."""
    g = git_metadata()
    if not allow_dirty:
        if g["code_commit"] is None:
            raise RuntimeError("no git metadata available; refusing to write a production result "
                               "(pass --allow-dirty to override)")
        if g["dirty_worktree"]:
            raise RuntimeError("worktree is dirty; commit before a production run "
                               "(pass --allow-dirty to override)")
    return {"schema_version": schema_version, "experiment_type": experiment_type,
            "estimand": estimand, "analysis_status": analysis_status, **g, **extra}


def require_schema(data: dict, *, expected_schema: str, expected_experiment: str,
                   allowed_estimands: set, allowed_statuses: set = frozenset({VALIDATED}),
                   source: str = "<file>") -> None:
    """Fail-fast admission check for an analysis reader. Raises ValueError with the offending file
    named, so a stale JSON sitting in the experiments dir can never slip into an aggregate."""
    got_schema = data.get("schema_version")
    if got_schema != expected_schema:
        raise ValueError(f"{source}: expected schema {expected_schema}, got {got_schema!r}. "
                         "Regenerate this result or pass the explicit legacy flag.")
    got_exp = data.get("experiment_type")
    if got_exp != expected_experiment:
        raise ValueError(f"{source}: expected experiment_type {expected_experiment!r}, got {got_exp!r}")
    got_est = data.get("estimand")
    if got_est not in allowed_estimands:
        raise ValueError(f"{source}: unapproved estimand {got_est!r} (allowed: {sorted(allowed_estimands)})")
    got_status = data.get("analysis_status")
    if got_status not in allowed_statuses:
        raise ValueError(f"{source}: analysis_status {got_status!r} not in {sorted(allowed_statuses)}. "
                         "Pass --include-legacy-absolute-patching / --include-exploratory-sweeps to admit it.")


def admits(data: dict, *, expected_schema: str, expected_experiment: str,
           allowed_estimands: set, allowed_statuses: set) -> bool:
    """Non-raising variant: True if the file is admissible under these rules."""
    try:
        require_schema(data, expected_schema=expected_schema, expected_experiment=expected_experiment,
                       allowed_estimands=allowed_estimands, allowed_statuses=allowed_statuses)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------------------
# Isolated production runs (audit r5 blocker #5). A shared output directory lets a stale but
# schema-compatible file from an earlier job silently enter a final report. Production runs get
# their own directory + manifest, and the analyzer validates the directory as a whole.
# --------------------------------------------------------------------------------------
ALPHA_RANGE = (0.25, 4.0)   # PREDEFINED norm-match admissibility band (audit r5 #6) -- never tuned post hoc


def new_run_dir(root: str, run_id: str) -> str:
    """experiments/<date>_<commit7>_<run_id>/ -- one directory per production run."""
    import datetime
    import os
    g = git_metadata()
    commit = (g["code_commit"] or "nocommit")[:7]
    # date is supplied by the caller's clock; kept in the name purely for human sorting
    stamp_date = datetime.date.today().isoformat()
    path = os.path.join(root, f"{stamp_date}_{commit}_{run_id}")
    os.makedirs(path, exist_ok=True)
    return path


def write_manifest(run_dir: str, *, run_id: str, schema_version: str, expected_models: list,
                   expected_experiments: list, expected_forms: list, allow_dirty: bool = False) -> dict:
    """Declare up-front what this run MUST produce, so a partial run cannot be silently analyzed."""
    import json
    import os
    g = git_metadata()
    if not allow_dirty and (g["code_commit"] is None or g["dirty_worktree"]):
        raise RuntimeError(f"production run requires a clean, known worktree (got {g}); "
                           "commit first or pass allow_dirty=True for a non-production run")
    man = {"run_id": run_id, "schema_version": schema_version, **g,
           "expected_models": list(expected_models),
           "expected_experiments": list(expected_experiments),
           "expected_forms": list(expected_forms),
           "allow_dirty": allow_dirty, "completion": {}}
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    return man


def validate_run_dir(run_dir: str, results: list, *, require_manifest: bool = True) -> dict:
    """Whole-directory admission check for a production analysis.

    `results` is [(basename, parsed_json), ...]. Rejects: mixed code commits, dirty-worktree outputs,
    duplicate (experiment, model) cells, and models/experiments the manifest expected but that are
    missing. Returns the manifest (or {} when not required). Raises ValueError on any violation.
    """
    import json
    import os
    mpath = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mpath):
        if require_manifest:
            raise ValueError(f"{run_dir}: no manifest.json -- production analyses must run against an "
                             "isolated run directory created by scripts/new_run.py")
        return {}
    man = json.load(open(mpath))

    commits, dirty, cells = set(), [], {}
    for name, d in results:
        c = d.get("code_commit")
        commits.add(c)
        if d.get("dirty_worktree"):
            dirty.append(name)
        key = (d.get("experiment_type"), d.get("model") or d.get("model_revision", {}).get("name"))
        cells.setdefault(key, []).append(name)

    if len(commits) > 1:
        raise ValueError(f"{run_dir}: results span MULTIPLE code commits {sorted(map(str, commits))} -- "
                         "a report must come from one build")
    if man.get("code_commit") and commits and man["code_commit"] not in commits:
        raise ValueError(f"{run_dir}: results commit {commits} != manifest commit {man['code_commit']}")
    if dirty and not man.get("allow_dirty"):
        raise ValueError(f"{run_dir}: dirty-worktree results present: {sorted(dirty)}")
    dupes = {k: v for k, v in cells.items() if len(v) > 1}
    if dupes:
        raise ValueError(f"{run_dir}: duplicate (experiment, model) cells: {dupes}")

    seen_models = {m for (_, m) in cells if m}
    missing = [m for m in man.get("expected_models", [])
               if not any(m.endswith(s) or s.endswith(m) or m == s for s in seen_models)]
    if missing:
        raise ValueError(f"{run_dir}: manifest expected models with no results: {missing} "
                         "(incomplete run -- do not report it)")
    return man
