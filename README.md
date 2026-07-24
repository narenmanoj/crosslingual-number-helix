# Cross-lingual number helix

**Do LLMs represent a number's *value* in one shared geometry regardless of how it is written — and
can that be shown causally?**

LLMs lay integers out on a Fourier "helix" and use it to do arithmetic
([Kantamneni & Tegmark, 2502.00873](https://arxiv.org/abs/2502.00873)). That was established on
English Arabic digits. This project asks whether the *same directions* carry `37`, `३७`, `٣٧`,
`thirty-seven`, and `treinta y siete` inside a single model — measured geometrically, then tested
**causally** by transporting a value from one surface form onto another and watching the arithmetic
move.

| | example | axis |
|---|---|---|
| Arabic digits | `37` | reference |
| Devanagari / Eastern-Arabic / fullwidth | `३७` `٣٧` `３７` | **script** (glyphs only) |
| English words | `thirty-seven` | **notation** (digit → word) |
| Spanish / French / German words | `treinta y siete` … | **language** |

---

## Status

**Methods: hardened and tested. Headline numbers: pending regeneration.**

The pipeline went through four rounds of external code audit. The last two rounds changed the
*estimand* (what the intervention actually manipulates), so effect sizes from earlier runs no longer
apply. Current results files are stamped `schema_version 2.2`; the analysis layer **refuses** older
or mismatched files rather than silently mixing them.

| | state |
|---|---|
| Core geometry (centered Fourier fit, held-out R², deterministic PCA) | ✅ validated, unit-tested |
| Primary causal estimand (matched-arithmetic delta transport) | ✅ implemented, reproduces on a live model |
| Statistical layer (clustered CIs, cluster permutation, BH-FDR, schema enforcement) | ✅ implemented + tested |
| Cross-model effect sizes and significance counts | ⏳ **awaiting the schema-2.2 rerun** |

Full history, per-model tables and the audit trail: **[progress.md](progress.md)**.

## What we have so far

Scoped to what the current code actually establishes:

- **Graded sharing (H2).** Using clean per-axis contrasts (script = `en_digit` vs digit-scripts,
  notation = `en_digit` vs `en_word`, language = `en_word` vs foreign words), **`language` is
  consistently the least-shared axis** across every model tested, well above a pipeline-matched
  permutation null. `script` and `notation` are both high; their relative order varies by model.
- **Exposure-dependent script sharing.** Cross-script sharing spans ~0.51→0.83 and tracks
  (multi)script training exposure — strongest in heavily multilingual models, weakest in
  English-primary ones. Observational and confounded by tokenizer/corpus/size; stated as a tendency.
- **Replication across model families.** The helix, cross-form sharing and causal transport appear in
  transformers *and* three hybrid state-space/attention models (Granite-4, Falcon-H1, Nemotron-Nano).
  We say "replicated across the tested families", **not** "architecture-independent" — 9 checkpoints
  across 7 orgs are not 9 independent architecture samples.
- **Causal sufficiency (direction robust).** Adding only the matched-arithmetic displacement
  `QQᵀ(h_en(a′,b) − h_en(a,b))` at a foreign-form number's token moves the answer toward `a′+b`,
  while norm-matched control subspaces do far less. Reproduces live; magnitudes pending rerun.
- **Causal necessity (weaker, script-biased).** Ablating the shared subspace hurts arithmetic more
  than a random subspace for most *interpretable* script cells, but the margin narrows against
  matched **structured** controls — the specific directions are partly redundant with other
  structured directions. Necessity for foreign number-*words* is largely undefined (base models are
  near floor on that task).

**A lesson from the hardened controls — the Haar control is not usable here.** Against a naive
Haar subspace the delta effect looks enormous, but a random 8-d subspace in ~1500-d captures only
≈√(8/1536) ≈ 7% of any vector, so norm-matching it needs α ≈ 8–10: the "matched" control is an
extrapolation, not a plausible intervention. This is structural, not bad luck — selecting the best 2
of 40 candidates still leaves α ≈ 10. The predefined admissibility band (0.25 ≤ α ≤ 4) therefore
**drops the Haar comparison from the primary analysis automatically**, and the defensible primary
controls are the structured ones, which *are* admissible: top-PCA-span (α ≈ 2) and shuffled-Fourier
(α ≈ 1.06). Against those the margin is smaller but honest. Haar remains available as a sensitivity
view via `--all-controls`.

## Reproducing

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --no-build-isolation -r requirements.txt   # --no-build-isolation for mamba-ssm
python tests/test_core.py                              # 30 unit tests, no model needed
```

Device auto-detects (CUDA → MPS → CPU). `Qwen/Qwen2.5-1.5B` smoke-tests on a laptop; the real runs
want a ≥24 GB GPU. **Causal legs require base (non-instruct) models** — instruct models score ~0 on
the bare `"a + b = "` readout.

### Single model, end to end

```bash
M=Qwen/Qwen2.5-7B; L=14        # L = the representational sharing peak for this model

# 1. geometry: helix fit + cross-form alignment (held-out R², coordinate cosines, permutation null)
python scripts/run_fit_and_align.py --model $M --layer $L

# 2. H2 (authoritative): clean per-axis contrasts + rank-aware overlap + pairwise matrix
python scripts/run_structure.py --model $M --layer $L

# 3. sufficiency: matched-arithmetic delta transport vs 3 norm-matched control families
python scripts/run_transport.py --model $M --layer $L --pairs-per-form 120

# 4. necessity: whole-span ablation (norm-matched controls, cross-fit baseline) + delta interchange
python scripts/run_necessity.py --model $M --layer $L --intervention-pos span \
    --pairs-per-form 120 --n-seeds 10

# 5. statistics over everything above (no model load, seconds)
python scripts/analyze_stats.py --out-dir experiments --b 20000
```

### Production runs (the only supported path for reportable numbers)

Production is a single manifest-driven command. It refuses a dirty worktree, creates an isolated run
directory, **freezes the causal layers** with the independent protocol, registers every expected
cell, records per-job success/failure, and analyzes *only* that directory:

```bash
RUN_ID=pilot01 MODELS="Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407" \
    bash scripts/run_overnight.sh
```

```
experiments/2026-07-23_<commit>_pilot01/
  manifest.json     expected cells, preregistered families, baseline policy, per-job completion
  layers.json       frozen layers + full per-layer held-out R² record
  transport_*.json  necessity_*.json      logs/run.log
```

The final `analyze_stats --production` **rejects the run** if anything is off: mixed commits, a dirty
result, a duplicate or missing cell, a file not in the manifest, an unfinished job, a legacy or
exploratory estimand, or any baseline fallback. A failed job therefore fails the *report* rather than
being papered over by a stale file.

Knobs: `MODELS`, `FORMS`, `POSITIONS` (default `last span after`), `PAIRS` (**0 = all valid triples**,
the production default), `CTRL_SEEDS`, `NSEEDS`, `RUN_SWEEPS=1` to add exploratory sweeps.

**Production vs scratch.** Production is the default and requires a clean worktree. `ALLOW_DIRTY=1`
declares a **scratch** run: results are explicitly not reportable, and the writers' production
contract is disabled (a layer manifest frozen from a dirty tree could never satisfy it).

**What counts as a positive result.** A cell is a headline positive only when it is FDR-significant
**and** its **crossed** interval — resampling both cases and the global control seeds — excludes zero.
Which control subspaces happened to be drawn is part of the scientific uncertainty, so the case-only
interval is reported as a conditional-on-this-bank diagnostic and never as the headline. Necessity
cells for forms the model cannot actually solve (clean accuracy below the preregistered threshold)
are labelled `not_testable_due_to_clean_behavior` and excluded from the primary family rather than
being counted as nulls.

Layers can also be frozen on their own:

```bash
python scripts/select_layers.py --models Qwen/Qwen2.5-7B --out layers.json
python scripts/run_transport.py --model Qwen/Qwen2.5-7B --layer-manifest layers.json --production
```

Runs transport → necessity → ablation-sweep per model, clears the HF cache between models, and runs
`analyze_stats.py` at the end. Knobs: `PAIRS` (cases/form, default 120), `NSEEDS`, `SWEEP_SEEDS`,
`STRIDE`, `CLEAN_CACHE`, `MODELS`, `OUT_DIR`.

> **Sample-size floor.** Inference clusters by source value, so a cell needs **≥ ~10 distinct source
> values** for the cluster permutation test to be able to reach significance at all. `PAIRS=120` is
> comfortable; tiny smoke runs will show wide CIs and no FDR hits by construction.

### Optional / opt-in analyses

```bash
python scripts/analyze_stats.py --include-exploratory-sweeps        # adds layer-sweep necessity peak
python scripts/analyze_stats.py --include-legacy-absolute-patching  # adds the legacy estimand
python scripts/analyze_stats.py --no-strict                         # warn-and-skip instead of failing
python scripts/aggregate_runs.py                                    # cross-model H2 (clean contrasts)
python scripts/run_ablation_sweep.py --model $M --layer-stride 2    # exploratory depth profile
```

## How the claims are kept honest

The design decisions that make the causal result defensible, all enforced in code:

- **One estimand per heading.** The primary sufficiency test transports *only* the value
  displacement between two matched English arithmetic prompts, holding addend, syntax, answer format
  and form/carrier offset fixed. The older "replace the subspace with a carrier reconstruction"
  intervention moved value *together with* context, and is now an opt-in `legacy_diagnostic`.
- **Controls that are actually matched.** Every control subspace is norm-matched per case, drawn from
  three families (Haar, top-PCA-span, shuffled-pipeline), and **every seed is retained** — so we can
  report *P(signal beats a random control draw)* and worst-control margins, not just "beats the mean".
  The norm-match scale α is stored **per (case, seed)** with a predefined admissibility band
  (0.25 ≤ α ≤ 4); the primary analysis keeps only admissible controls and `--all-controls` gives the
  sensitivity view. Because a naive Haar subspace in ~1500-d captures almost none of an 8-d
  displacement (α ≈ 8, inadmissible), control banks are **energy-matched**: we draw a larger candidate
  pool and keep the subspaces whose *natural* projected energy resembles the helix's. The selection is
  recorded in the output, not hidden.
- **The subspace never sees the values it is tested on.** Q is fitted on numbers 10–99 while the
  causal test uses 0–9, so "the intervention works" cannot be an artifact of fitting the exact test
  values. `value_sets_disjoint` is stamped in every result.
- **Layer choice is frozen before looking.** The layer is picked from **en_digit only**, on a
  disjoint discovery split, scoring **held-out** R² — never from the cross-form comparison it will
  later support. Production consumes a commit-stamped `layers.json`; a hand-typed `--layer` is
  refused outright.
- **Admissible controls decide the headline number.** Control-seed admissibility is a property of the
  *global seed* (one seed = one subspace reused across all cases), so the primary point estimate, CI,
  permutation p and FDR are all recomputed from the admitted seeds. Whole seeds are dropped — never
  imputed — and a cell **fails** rather than reports if too few seeds survive.
- **Multiple-testing families are preregistered** in `config.py` and copied into each run's manifest
  before the run starts, so they cannot be chosen after seeing results.
- **Fail-fast provenance.** Every result stamps schema / experiment type / estimand / analysis status
  / git commit / worktree state; every analysis validates them and **refuses** stale or mismatched
  files. Exploratory sweep results are excluded from the default statistics unless explicitly opted in.
- **Consistent inference.** Paired differences are matched strictly by case key (never by position);
  CIs are cluster bootstraps over source value; the permutation test flips signs at the cluster level
  so the p-value and CI assume the same independent unit; BH-FDR runs over the validated family only.
- **Hook equivalence is verified, not assumed.** Before any patch, the script asserts the forward hook
  sees exactly `hidden_states[L]` for that architecture — load-bearing for the hybrid/SSM models.

## Repo layout

```
config.py                    defaults + SCHEMA_VERSION (bump on any estimand change)
src/data.py                  surface forms + carrier prompts (script / notation / language axes)
src/extract.py               model loading, activation extraction, fail-fast answer-token ids
src/helix.py                 centered Fourier fit, held-out R², shuffled-label control
src/alignment.py             principal angles, rank-aware overlap, coordinate cosines, permutation null
src/patching.py              subspace/delta patches, norm-matched controls, hook-equivalence check
src/provenance.py            schema stamping + fail-fast admission gate
scripts/run_*.py             the experiments (see "Reproducing")
scripts/analyze_stats.py     estimand-aware statistics -> stats_summary.json + stats_forest.png
scripts/run_overnight.sh     unattended multi-model driver
tests/test_core.py           30 unit tests (no model required)
experiments/                 result JSONs + figures
```

## Limitations

The task is **single-digit addition** with **Arabic-digit answers**, so the mechanism may be a shared
digit-normalized arithmetic *interface* rather than format-independent quantity. Script sharing may
reflect common positional-decimal parsing rather than abstract value. The language contrast varies
both the number word and its carrier sentence (a factorial design is still to run). Downstream
*sensitivity*-matched controls remain open — norm matching is not sensitivity matching. No temporal
"read-layer" claim is made; the depth analysis is exploratory only. See
[progress.md](progress.md) for the full threat list and the open-items queue.

## Key references

- Kantamneni & Tegmark, *LLMs Use Trigonometry to Do Addition* — [2502.00873](https://arxiv.org/abs/2502.00873)
- Engels et al., *Not All Features Are One-Dimensionally Linear* — [2405.14860](https://arxiv.org/abs/2405.14860)
- Makelov et al., *An Interpretability Illusion for Subspace Activation Patching* — [2311.17030](https://arxiv.org/abs/2311.17030)
- Wendler et al., *Do Llamas Work in English?* — [2402.10588](https://arxiv.org/abs/2402.10588)
- *The Semantic Hub Hypothesis* — [2411.04986](https://arxiv.org/abs/2411.04986)
