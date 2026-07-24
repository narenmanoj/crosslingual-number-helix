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

import os
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


def resolve_layer(model: str, layer_arg, manifest_path: str = None, schema_version: str = None,
                  production: bool = False) -> tuple:
    """Resolve the causal layer, preferring a FROZEN layer manifest (audit r6 blocker #7).

    In production a hand-typed --layer is refused: the layer must come from a manifest produced by
    scripts/select_layers.py at THIS commit, using the approved independent protocol. Returns
    (layer, provenance-dict)."""
    import json
    if manifest_path:
        man = json.load(open(manifest_path))
        if schema_version and man.get("schema_version") != schema_version:
            raise ValueError(f"{manifest_path}: schema {man.get('schema_version')} != {schema_version}")
        if man.get("selection_protocol") != "en_digit_heldout_r2":
            raise ValueError(f"{manifest_path}: unapproved selection protocol {man.get('selection_protocol')!r}")
        if production:
            g = git_metadata()
            if man.get("dirty_worktree"):
                raise ValueError(f"{manifest_path}: layers were frozen from a dirty worktree")
            if g["code_commit"] and man.get("code_commit") != g["code_commit"]:
                raise ValueError(f"{manifest_path}: frozen at commit {man.get('code_commit')}, "
                                 f"running at {g['code_commit']} -- re-freeze layers for this build")
        entry = (man.get("models") or {}).get(model)
        if entry is None:
            raise ValueError(f"{manifest_path}: no frozen layer for {model!r}")
        frozen_rev = model_commit(entry)          # revision -> commit_hash -> content_hash (r9 #4)
        if production and frozen_rev is None:
            raise ValueError(f"{manifest_path}: no immutable model revision for {model!r} -- "
                             "production requires a pinned snapshot")
        return int(entry["selected_layer"]), {
            "layer_source": "frozen_manifest", "layer_manifest": os.path.basename(manifest_path),
            "selection_protocol": man.get("selection_protocol"),
            "discovery_numbers": man.get("discovery_numbers"),
            "evaluation_numbers": man.get("evaluation_numbers"),
            "selection_frozen_before_crossform_evaluation": True,
            "frozen_model_revision": frozen_rev,   # r8 #7 / r9 #4
            "manifest_commit": man.get("code_commit")}
    if production:
        raise ValueError("production runs require --layer-manifest (hand-picked layers are not "
                         "reproducible; generate one with scripts/select_layers.py)")
    if layer_arg is None:
        raise ValueError("no --layer and no --layer-manifest given")
    return int(layer_arg), {"layer_source": "cli_argument",
                            "selection_frozen_before_crossform_evaluation": False}


def result_cell_id(d: dict) -> tuple:
    """FULL identity of a result cell (audit r6 blocker #4).

    (experiment_type, model) is too coarse: necessity at last/span/after for one model are three
    legitimate cells, not duplicates. Position and estimand are part of the identity."""
    return (d.get("experiment_type"),
            d.get("model") or (d.get("model_revision") or {}).get("name"),
            d.get("estimand"),
            d.get("layer"),
            d.get("pooling"),
            d.get("ablation_position"),
            d.get("interchange_position"))


CELL_FIELDS = ["experiment_type", "model", "estimand", "layer", "pooling",
               "ablation_position", "interchange_position"]


def cell_matches(expected: dict, cell: tuple) -> bool:
    """An expected-cell spec matches an observed cell on the keys it actually specifies."""
    for i, f in enumerate(CELL_FIELDS):
        if f in expected and expected[f] is not None and expected[f] != cell[i]:
            return False
    return True


def model_commit(d: dict):
    """The immutable model snapshot for a result file (audit r8 #7, r9 #4): the pinned HF revision,
    else the resolved config commit hash, else (local models) a recorded content hash. NOT the code
    commit. None when nothing immutable was recorded -- which production treats as a failure."""
    mr = d.get("model_revision") or {}
    return mr.get("revision") or mr.get("commit_hash") or mr.get("content_hash")


def default_analysis_policy(**over) -> dict:
    """FROZEN analysis policy (audit r7 #8, r8 #5). EVERY choice that can change which cells are
    included or called significant lives here, is written into the run manifest BEFORE the run, and is
    enforced in production -- so a report cannot be re-derived with friendlier settings."""
    import config as C
    pol = {"necessity_null": "shuf_fourier",           # r8 #5
           "admissible_only": True,                    # r8 #5
           "fdr_alpha": 0.05,                          # r8 #5
           "alpha_range": list(ALPHA_RANGE),
           "min_case_fraction": 1.0,                   # r8 #9: STRICT -- every case admissible
           "min_admitted_seeds": 5,
           "cluster_by": 0,                            # 0 = source value
           "bootstrap_B": 20000,
           "primary_requires_crossed_ci": True,        # r7 #3
           "primary_necessity_position": C.PRIMARY_NECESSITY_POSITION,   # r8 #4
           "secondary_necessity_positions": list(C.SECONDARY_NECESSITY_POSITIONS),
           "global_fdr_sensitivity": True,
           "clean_accuracy_threshold": 0.8,            # r7 #9 (necessity eligibility)
           "baseline_policy": "in_run_leave_one_source_value_out",   # r8 #1: honest label
           "require_all_cases_processed": True}        # r7 #10
    pol.update(over)
    return pol


def write_manifest(run_dir: str, *, run_id: str, schema_version: str, expected_models: list,
                   expected_experiments: list, expected_forms: list, expected_cells: list = None,
                   primary_families: list = None, secondary_families: list = None,
                   baseline_policy: str = "in_run_leave_one_source_value_out", required_fallback_count: int = 0,
                   analysis_policy: dict = None, allow_dirty: bool = False,
                   necessity_ineligible_forms: dict = None) -> dict:
    """Declare up-front what this run MUST produce, so a partial run cannot be silently analyzed.

    `expected_cells` enumerates the EXACT cells (including intervention positions) the run must emit;
    `primary_families` / `secondary_families` preregister the multiple-testing families (audit r6 #11);
    `required_fallback_count` pins the baseline policy (audit r6 #6)."""
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
           "expected_cells": list(expected_cells or []),
           "primary_hypothesis_families": list(primary_families or []),
           "secondary_families": list(secondary_families or []),
           "global_fdr_sensitivity": True,
           "baseline_policy": baseline_policy,
           "required_fallback_count": required_fallback_count,
           "necessity_ineligible_forms": dict(necessity_ineligible_forms or {}),   # r8 #2
           "analysis_policy": analysis_policy or default_analysis_policy(baseline_policy=baseline_policy),
           "allow_dirty": allow_dirty, "completion": {}}
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    return man


def record_completion(run_dir: str, job_id: str, status: str, detail: str = "") -> None:
    """Mark one expected job succeeded/failed so an incomplete run cannot masquerade as complete."""
    import json
    import os
    mpath = os.path.join(run_dir, "manifest.json")
    man = json.load(open(mpath))
    man.setdefault("completion", {})[job_id] = {"status": status, "detail": detail}
    with open(mpath, "w") as fh:
        json.dump(man, fh, indent=2)


def validate_run_dir(run_dir: str, results: list, *, require_manifest: bool = True,
                     strict_cells: bool = True) -> dict:
    """Whole-directory admission check for a production analysis (audit r6 blockers #3/#4/#6).

    `results` is [(basename, parsed_json), ...]. Enforces: one code commit matching the manifest, a
    clean worktree, exact schema, no duplicate FULL cells, every expected cell present, no unexpected
    files, expected forms present inside each file, job completion, model revision recorded, no
    legacy/exploratory estimand, and zero baseline fallbacks/skips. Raises ValueError on violation.
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

    commits, dirty, cells, problems = set(), [], {}, []
    for name, d in results:
        commits.add(d.get("code_commit"))
        if d.get("dirty_worktree"):
            dirty.append(name)
        cells.setdefault(result_cell_id(d), []).append(name)
        if d.get("schema_version") != man.get("schema_version"):
            problems.append(f"{name}: schema {d.get('schema_version')} != manifest {man.get('schema_version')}")
        if d.get("analysis_status") not in (None, VALIDATED):
            problems.append(f"{name}: {d.get('analysis_status')} result in a validated production report")
        # model snapshot must be PINNED, not just present (audit r8 #7)
        if not d.get("model_revision"):
            problems.append(f"{name}: no model_revision recorded")
        elif model_commit(d) is None:
            problems.append(f"{name}: model_revision has no revision/commit hash (weights not pinned)")
        # the manifest's baseline policy must match what the writer actually did (audit r8 #1)
        wpol = d.get("baseline_policy") or d.get("baseline_fit_split")
        if d.get("experiment_type") == "necessity" and wpol and man.get("baseline_policy") \
                and wpol != man["baseline_policy"]:
            problems.append(f"{name}: baseline_policy {wpol!r} != manifest {man['baseline_policy']!r}")
        # exhaustive matched-delta coverage: delta keys must equal the expected case set (audit r8 #8)
        for form, R in (d.get("results") or {}).items():
            keys = R.get("per_case_keys", {})
            exp = R.get("expected_case_keys")
            dk = keys.get("delta")
            if exp is not None and dk is not None and \
                    {tuple(k) for k in dk} != {tuple(k) for k in exp}:
                problems.append(f"{name}[{form}]: {len(exp) - len(dk)} case(s) missing from the "
                                "matched-delta arrays (incomplete en_digit reference coverage)")
        # r9 #6: a necessity form REGISTERED as eligible that reads below threshold in the RESULT is an
        # UNEXPECTED failure (revision/tokenizer/nondeterminism), NOT a preregistered not-testable form.
        if d.get("experiment_type") == "necessity":
            thr = (man.get("analysis_policy") or {}).get("clean_accuracy_threshold", 0.8)
            ineligible = set(man.get("necessity_ineligible_forms") or {})
            for form, A in (d.get("ablation") or {}).items():
                key = f"{d.get('model')}:{form}"
                acc = A.get("clean_acc")
                if acc is not None and acc < thr and key not in ineligible:
                    problems.append(f"{name}[{form}]: clean_acc {acc:.2f} < {thr} but was registered "
                                    "ELIGIBLE -- unexpected behavioural failure (not preregistered)")
        # r9 #8: writer metadata must match the frozen experiment policy
        ep = man.get("experiment_policy") or {}
        if ep:
            checks = {"fit_values": d.get("fit_values"), "causal_values": d.get("causal_values")}
            for k, got in checks.items():
                if got is not None and ep.get(k) is not None and list(got) != list(ep[k]):
                    problems.append(f"{name}: {k} {got} != frozen experiment_policy {ep[k]}")
            if d.get("case_set_exhaustive") is False and ep.get("exhaustive_cases"):
                problems.append(f"{name}: sampled cases but experiment_policy requires exhaustive")
        # zero-fallback baseline policy (r6 blocker #6)
        for form, A in (d.get("ablation") or {}).items():
            if A.get("n_skipped_no_baseline"):
                problems.append(f"{name}[{form}]: {A['n_skipped_no_baseline']} case(s) skipped for want of a baseline")
            for pm in (A.get("baseline_meta") or []):
                if any(v.get("fallback_used") for v in pm.values()):
                    problems.append(f"{name}[{form}]: baseline fallback used")
                    break
        # NON-EMPTY payload (r7 #5). Per-cell form validation happens after 1:1 matching below (r9 #1):
        # the manifest-level union must NOT be applied to every file, or a necessity file (fewer forms)
        # is wrongly rejected against the transport form set.
        payload = d.get("results") or d.get("ablation") or {}
        if not payload:
            problems.append(f"{name}: empty results/ablation payload")
        # every present form must have actually processed cases (r7 blockers #5/#10)
        for form, blk in payload.items():
            n_cases = blk.get("n_cases", blk.get("n"))
            if n_cases is not None and n_cases <= 0:
                problems.append(f"{name}[{form}]: zero processed cases")
            if d.get("all_cases_processed") is False:
                problems.append(f"{name}[{form}]: selected cases != processed cases "
                                f"(skipped: {d.get('skipped_case_keys')})")

    if len(commits) > 1:
        problems.append(f"results span MULTIPLE code commits {sorted(map(str, commits))}")
    if man.get("code_commit") and commits and man["code_commit"] not in commits:
        problems.append(f"results commit {commits} != manifest commit {man['code_commit']}")
    if dirty and not man.get("allow_dirty"):
        problems.append(f"dirty-worktree results present: {sorted(dirty)}")
    dupes = {str(k): v for k, v in cells.items() if len(v) > 1}
    if dupes:
        problems.append(f"duplicate cells: {dupes}")

    # every result must pin the SAME model snapshot for a given model (audit r8 #7)
    by_model = {}
    for name, d in results:
        m = d.get("model") or (d.get("model_revision") or {}).get("name")
        by_model.setdefault(m, set()).add(model_commit(d))
    for m, revs in by_model.items():
        real = {r for r in revs if r is not None}
        if len(real) > 1:
            problems.append(f"{m}: results used DIFFERENT model revisions {sorted(real)} "
                            "(weights/tokenizer changed between jobs)")

    # layer-manifest revision must equal what every job actually loaded (r9 #4)
    layer_rev = None
    lpath = os.path.join(run_dir, "layers.json")
    if os.path.exists(lpath):
        lm = json.load(open(lpath))
        revs = {(v.get("model_revision") or {}).get("revision") for v in (lm.get("models") or {}).values()}
        for name, d in results:
            m = d.get("model") or (d.get("model_revision") or {}).get("name")
            lr = ((lm.get("models") or {}).get(m) or {}).get("model_revision", {}).get("revision")
            if lr is not None and model_commit(d) is not None and lr != model_commit(d):
                problems.append(f"{name}: model revision {model_commit(d)} != layer-manifest {lr}")

    expected = man.get("expected_cells") or []
    obs_by_name = {result_cell_id(d): (nm, d) for nm, d in results}
    if expected and strict_cells:
        observed = list(cells)
        # ONE-TO-ONE assignment (r8 #6): every expected cell matches exactly one observed cell and
        # vice versa. Then per-cell FORM validation and required-secondary completeness (r9 #1/#7).
        for spec in expected:
            hits = [c for c in observed if cell_matches(spec, c)]
            if len(hits) == 0:
                if spec.get("requirement", "required_primary") != "optional":
                    problems.append(f"expected cell missing: {spec}")
            elif len(hits) > 1:
                problems.append(f"expected cell matched by {len(hits)} results (under-specified?): "
                                f"{spec} -> {hits}")
            else:
                nm, d = obs_by_name.get(hits[0], (None, None))
                if d is not None and spec.get("expected_forms") is not None:
                    got = set((d.get("results") or d.get("ablation") or {}).keys())
                    want = set(spec["expected_forms"])
                    if got != want:
                        problems.append(f"{nm}: payload forms {sorted(got)} != expected "
                                        f"{sorted(want)} for cell {spec.get('experiment_type')}"
                                        f"@{spec.get('ablation_position', '')}")
        for c in observed:
            hits = [spec for spec in expected if cell_matches(spec, c)]
            if len(hits) == 0:
                problems.append(f"unexpected result cell not in manifest: {c}")
            elif len(hits) > 1:
                problems.append(f"result cell matches {len(hits)} expected specs (ambiguous): {c}")
    elif not expected:
        seen_models = {c[1] for c in cells if c[1]}
        missing = [m for m in man.get("expected_models", []) if m not in seen_models]
        if missing:
            problems.append(f"manifest expected models with no results: {missing}")

    # the manifest must not promise experiments the run never registered (r7 blocker #7)
    if expected:
        declared = set(man.get("expected_experiments") or [])
        in_cells = {c.get("experiment_type") for c in expected}
        if declared and declared != in_cells:
            problems.append(f"manifest expected_experiments {sorted(declared)} != experiment types in "
                            f"expected_cells {sorted(in_cells)} -- declare the run's actual scope")

    incomplete = {k: v for k, v in (man.get("completion") or {}).items() if v.get("status") != "ok"}
    if incomplete:
        problems.append(f"jobs not completed successfully: {sorted(incomplete)}")
    if not (man.get("completion") or {}):
        problems.append("no job completion records -- the runner did not register its jobs")

    if problems:
        raise ValueError(f"{run_dir}: " + "; ".join(problems))
    return man
