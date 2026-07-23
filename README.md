# Cross-lingual number helix

**Do large language models share a single, language-agnostic geometry of quantity —
and can we prove it causally?**

This is a mechanistic-interpretability research project aimed at an ML-conference paper
(workshop as a first milestone). This README explains the idea, why it's worth doing, the
specific hypothesis, and how the code tests it.

## Summary of findings (working thesis)

> **Number geometry is *partially* universal across surface forms.** LLMs represent integers on a
> Fourier "helix"; that helix subspace is **shared** across scripts, notations, and languages —
> but only *partially*, and the degree of sharing degrades as the surface transformation grows.

Thesis: **partial geometric universality, heterogeneous computational use.** Evidence (**9 base
causal models across 7 orgs + 3 non-transformer architectures**, plus 1 instruct model
representational-only; all causal claims now carry **bootstrap 95% CIs + paired significance**):

- **Graded sharing (H2):** with clean per-axis contrasts, `language` is the **consistently
  least-shared** axis, far above a random floor, in every model tested. `script` and `notation` are
  both high but their *relative* order varies by model. Magnitude varies; the language-is-lowest
  ordering doesn't.
- **Exposure-dependent script sharing:** cross-*script* sharing spans **0.51→0.83** and tracks
  (multi)script training — multilingual models (Qwen3 119-lang 0.83, Mistral 0.80, Qwen2.5 0.77,
  Granite 0.76) share strongly; English-primary OLMo-3 is lowest (0.51). (Aya, multilingual but 0.53,
  is a caveat — likely *numeral-script* exposure specifically, not general multilinguality.)
- **Replicated across transformer *and* hybrid families:** the helix, cross-form sharing, and causal
  transport all appear in the three **non-transformers** — Granite-4 (hybrid Mamba-2/MoE), Falcon-H1
  (parallel hybrid), and Nemotron-Nano (Mamba-2 hybrid). We say *replicated across the tested
  transformer and hybrid state-space/attention families* rather than "architecture-independent" — 9
  checkpoints across 7 orgs are not 9 independent architecture samples (audit #13); report at
  checkpoint, family, and org granularity.
- **Causally *sufficient* everywhere — the strong universal claim.** Injecting the shared subspace
  (subspace-only) with a value steers arithmetic for *every* form, while a **norm-matched** random
  subspace does essentially nothing: **45/45 model×form cells significant** (95% CI excludes 0),
  across all 9 models and every script/notation/language axis. Matched-source interchange (real
  activation, subspace-only, vs norm-matched random) is likewise **45/45**.
- **Causally *necessary* — real but weaker, script-specific, and depth-dependent.** Two instruments:
  (i) **vs a random null across depth** (layer sweep, held-out): significant for **~80% of
  *interpretable* script cells** (clean-acc ≥ 0.6) but ~0% for spelled-out words — and necessity
  peaks *shallower* (L1–L7) than the representational sharing layer. (ii) **vs a matched *structured*
  null** (shuffled-Fourier) the margin shrinks: the shared directions are partly **redundant** with
  other structured directions. Necessity for foreign number-*words* is largely undefined — base
  models often can't do the arithmetic there to begin with (floor clean-acc).

**Net:** number geometry is partially shared across scripts, notations, and languages, in
transformers *and* three Mamba/hybrid models; the shared directions are **causally sufficient** to
drive arithmetic in every form (universal, 45/45 with CIs), while **reliance** is helix-specific but
**weaker, script-biased, and shallower** than the sharing layer. Sharing is graded (language always
lowest), and **cross-script sharing tracks training exposure**. **Causal results need base models;
Aya (instruct) is representational-only.** We make *no* temporal "read-layer" claim — see
[Limitations](#limitations--planned-strengthening).

---

## The idea in one minute

Recent interpretability work found that LLMs don't store numbers as arbitrary token
embeddings — they lay them out on a **generalized helix**. Concretely, the residual-stream
representation of an integer `n` is well fit by a few Fourier features (cosines/sines at
periods `[2, 5, 10, 100]`) plus a linear term, and this helix is **causally** used to do
arithmetic: patching along it changes the model's answer
([Kantamneni & Tegmark, 2502.00873](https://arxiv.org/abs/2502.00873);
[Engels et al., 2405.14860](https://arxiv.org/abs/2405.14860)).

But every result establishing this was measured on **English Arabic digits** — the token
`37`. A number, though, has many surface forms:

| | example | |
|---|---|---|
| Arabic digits | `37` | (English) |
| Devanagari digits | `३७` | (same language, different script) |
| Eastern-Arabic digits | `٣٧` | (same language, different script) |
| English words | `thirty-seven` | (different surface form) |
| Spanish words | `treinta y siete` | (different language) |
| French / German words | `trente-sept` / `siebenunddreißig` | (different language) |

**The question this project asks:** is "thirty-seven-ness" *one* geometric object inside the
model — the same helix, in the same directions, regardless of how the quantity was written —
or does each language/script get its own private representation that only gets reconciled
later? And critically, can we show the answer **causally**, not just correlationally: fit the
helix from one form and *transport* a number expressed in another form onto it, then watch the
model's arithmetic behave as if it had seen the transported value?

## Why this matters

- **A real question about LLM cognition.** "Does the model have a unified concept of *quantity*,
  or does it think about numbers per-language?" is a concrete, falsifiable version of the
  bigger question of whether LLMs build language-agnostic world models.
- **Safety / reliability angle.** LLM arithmetic accuracy is known to *drop* on
  under-represented scripts and spelled-out forms even though the math is identical
  ([2601.15251](https://arxiv.org/abs/2601.15251)). If the geometry is *not* shared, that's a
  mechanistic explanation for the failure — and a lever to fix it.
- **It's a clean causal-interpretability story.** The field has moved past "we found a probe";
  novelty now lives in *causal* claims with proper controls. This project is built around a
  causal intervention from day one.

## The central hypothesis (stated as testable predictions)

> **H1 (shared geometry).** Across surface forms, the number helix occupies the *same
> directions* in the residual stream — so the subspaces align far above a random-subspace floor.
>
> **H2 (graded coordinate invariance).** Shared Fourier coordinates survive glyph-only changes
> (`37`↔`३७`) more completely than digit→word changes (`37`↔`thirty-seven`), which survive more than
> language changes (`thirty-seven`↔`treinta y siete`). *This is a statement about the geometry, not
> the mechanism.* High alignment across decimal digit-scripts need not mean an abstract "value" code:
> all four digit-scripts share the **same positional-decimal composition rule**, so the overlap could
> equally reflect a shared **glyph-normalization / digit-parsing** stage. Whether it is abstract
> value, common decimal parsing, or a shared digit-normalization stage is an *interpretation left to
> test* — H2 claims only the graded invariance.
>
> **H3 (causal transport & reliance).** A number written in form B can be activation-patched onto
> the form-A helix and shift the model's downstream arithmetic toward that value (sufficiency), and
> ablating the shared subspace can disrupt arithmetic in form B (necessity) — both against matched
> controls. High subspace overlap makes transport *geometrically plausible*; the causal test is what
> shows the downstream computation actually interprets the transported coordinates.

Any clear outcome is a paper. **Even a clean negative** — "the helix is *not* shared across
forms, and here's the tokenization-controlled evidence" — is publishable if pre-registered,
because it contradicts the implicit universality assumption in the number-geometry literature.

## What's novel (narrow, and honestly bounded)

This is a **specific-combination** contribution, not a new phenomenon or a new method. The
neighborhood is well-populated (all citations below verified 2026-07-18):

- **The helix substrate.** Numbers occupy a Fourier/helix code, causally used for addition
  (Kantamneni & Tegmark, [2502.00873](https://arxiv.org/abs/2502.00873)) — but **digit-only, English**.
  This is what we test the cross-form invariance *of*.
- **Cross-script helix *shape* is already observed — descriptively.** A June 2026 analysis (Gupta,
  *[From Latin Digits to Babylonian Cuneiform](https://girishgupta.com/beyond-the-parrot/20260618-from-latin-digits-to-babylonian-cuneiform)*)
  fits the helix across **10 numeral systems × 8 models** and reports glyph-invariant, value-driven,
  base-retuning helices — but **passively**: no arithmetic, no causal test, **no shared-subspace-overlap
  metric** (coincident *independent* per-script fits, not measured alignment), and no spelled-out
  number-words. → This pre-empts our cross-script *descriptive* claim; we do **not** headline it.
- **Digits and number-words share machinery — at the circuit level.** Lan, Torr & Barez
  ([2311.04131](https://arxiv.org/abs/2311.04131), EMNLP 2024) show shared *circuits* (heads) across
  numerals / number-words / months, causally necessary via ablation, incl. **English/Spanish** and
  addition/subtraction — but **head-level, Latin-script only, no helix/subspace geometry, no transport**.
  Semantic Hub ([2411.04986](https://arxiv.org/abs/2411.04986)) shows "5+3"/"five plus three" share an
  English-anchored space with causal steering — coarse, not helix-coordinate.
- **Format-agnostic subspaces + patching + ablation are an established *method*.** FARS
  ([2605.09496](https://arxiv.org/abs/2605.09496)) does PCA-subspace + activation patching + ablation +
  cross-architecture CCA — but for **reasoning concepts across prose/code/math**, *not* numbers, *not*
  the Fourier helix, *not* numeral scripts, *not* number-words. **This is our closest methodological
  prior;** we differ in **object** (number values), **coordinate system** (the specific Fourier-helix
  directions, not variance-PCA), and **forms** (scripts/notation/languages).
- Number-code universality is established **across models** ([2510.26285](https://arxiv.org/abs/2510.26285),
  [2604.20817](https://arxiv.org/abs/2604.20817)) — orthogonal to our **cross-form, within-model** question.

**Our defensible contribution** — the intersection none of the above occupies:
> Do the *Fourier-helix number directions* **share a subspace** across **numeral scripts** (Devanagari
> / Arabic-Indic / fullwidth), **notation** (digit↔word), and **language** (EN/ES/FR/DE number-words)
> *within a single model*; do those directions support **causal cross-form arithmetic transport**
> (natural-activation interchange + matched-arithmetic delta transport); are they **naturally
> necessary** (matched-null ablation); and how does this vary **by architecture**?

> **Span vs coordinates (precision, per code audit #1).** Principal-angle `subspace_cos` measures a
> shared *span* — two forms could share the 8-dim helix subspace while a rotation relabels which
> direction is period-2 vs the linear term. We therefore also report **coordinate-level identity**
> (`canonical_map_cosines`: per-feature signed cosine between the two forms' feature→model maps) and
> a **pipeline-matched permutation null** (`permutation_alignment_null`), and we say "shared subspace"
> for the span result and "shared coordinates" only where the canonical cosines are high.

Three load-bearing elements — drop any one and a prior paper covers us: **helix-coordinate /
principal-angle** (else → FARS), **numeral-script** (else → Semantic Hub / script-invariance SAEs),
**within-model cross-form** (else → the cross-model universality work). We position this as *the
cross-form companion to the cross-model universality results, specialized to the K–T helix, with
FARS-style causal methodology applied where FARS does not go* — and we pre-empt the reviewer's
"isn't this FARS/Semantic Hub for numbers?" head-on.

The contribution is to connect these: take the *causally-validated helix* and the
*causally-validated cross-lingual transport method* and answer the question both literatures
skipped.

**Closest adjacent work to differentiate from** (novelty recheck, mid-2026): *FARS —
Format-Agnostic Reasoning Subspaces* ([2605.09496](https://arxiv.org/abs/2605.09496)) does
cross-form activation patching across languages/symbolic forms, but for **general reasoning
concepts via a generic PCA subspace**, not the **Fourier number helix**, and **without non-Latin
numeral scripts**. Our contribution is the *object* (the specific helix) + the *cross-script*
numeral dimension + causal transport — framed as "new object + new dimension," **not** "new
method." The universal-numbers paper ([2510.26285](https://arxiv.org/abs/2510.26285)) explicitly
disclaims cross-lingual coverage — it names our gap.

---

## How the code tests it

```
config.py                     # defaults (model, numbers, layer, ...) — override on CLI
src/data.py                   # renders numbers across the two axes (script vs language)
src/extract.py                # loads a model, pulls the residual-stream vector at the number token
src/helix.py                  # fits the helix (PCA + Fourier), R^2, shuffled-label control
src/alignment.py              # subspace principal angles + Procrustes-CV + CKA + random floor
src/patching.py               # STEP-3 machinery: helix reconstruct/subspace + full/subspace/random patch
scripts/run_fit_and_align.py  # steps 1+2: fit + cross-form alignment, single layer
scripts/run_layer_sweep.py    # fit+align at EVERY layer -> subspace_cos-vs-layer plot per axis
scripts/run_transport.py      # STEP 3 sufficiency: causal cross-form transport + full/subspace/random controls
scripts/run_necessity.py      # STEP 3 necessity (single layer): ablation + matched-source interchange
scripts/run_ablation_sweep.py    # necessity vs layer: helix-vs-random Δ per layer + helix-vs-STRUCTURED nulls at the peak (held-out CIs)
scripts/run_transport_sweep.py   # transport at every layer (layer-normalized subspace/full)
scripts/run_structure.py      # #6 pairwise form x form matrix + #7 geometry<->behavior (one model load)
scripts/aggregate_runs.py     # collect experiments/align_*.json -> cross-model table + bar chart
scripts/analyze_stats.py      # bootstrap 95% CIs + paired significance tests over the per-case logs (no model load)
scripts/run_overnight.sh      # unattended full-model loop -> per-case logs -> analyze_stats (per-model cache cleanup)
scripts/inspect_tokenization.py  # diagnostic: token counts + what each pooling reads per form
```

**Three axes of variation** (`src/data.py`), ordered by increasing surface-form distance. All forms
render the *same integer set in the same order*, so activation rows are paired across forms (this is
what makes the comparisons valid):
- **Script axis** (same language + notation, only glyphs change):
  `en_digit` `37` vs `devanagari_digit` `३७` vs `arabic_indic_digit` `٣٧` vs `fullwidth_digit` `３７`.
  *Note: the cross-script helix **shape** was shown descriptively (and across more scripts) by Gupta
  (2026); our contribution on this axis is the **measured subspace overlap** and the **causal**
  (transport + necessity) test, not the descriptive observation. High sharing here shows
  glyph-invariance; it doesn't by itself isolate "value" from shared positional-decimal notation.*
- **Notation axis** (digits vs spelled-out words, language fixed): `en_digit` vs `en_word`.
- **Language axis** (spelled-out words, language varies): `en_word`, `es_word`, `fr_word`, `de_word`.

**H2 prediction (graded invariance):** compatibility should fall `script ≥ notation ≥ language`.
`run_fit_and_align.py` prints this per-axis summary; `run_structure.py` reports the *clean* contrasts
(correct reference per axis) + the full pairwise matrix, which is the primary object.

### Setup

```bash
cd /Users/nsm/LLM/crosslingual-number-helix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Device auto-detects (CUDA → MPS → CPU). Default model `Qwen/Qwen2.5-1.5B` smoke-tests on a
laptop; use the cluster for 7–9B (`Qwen2.5-7B`, `Llama-3.1-8B`, `aya-23-8B`, `gemma-2-9b`).
Base models are preferred over chat-tuned ones for cleaner number geometry.

### Run the first experiment

```bash
# fast iteration (default small model)
python scripts/run_fit_and_align.py

# real run on the cluster
python scripts/run_fit_and_align.py --model Qwen/Qwen2.5-7B \
    --forms en_digit en_word devanagari_digit es_word fr_word de_word
```

It (1) extracts the residual-stream vector at each number's last token for every form,
(2) fits the helix per form and reports R² (with a shuffled-label control that should collapse
to ~0), then (3) reports three alignment metrics vs the `en_digit` reference against a
**random-subspace floor**, and saves JSON to `experiments/`.

### Significance runs (bootstrap CIs + paired tests)

The causal legs log per-case outcomes, so significance is a separate, model-free step:

```bash
# unattended: every base model -> transport + necessity(span) + ablation-sweep -> stats
bash scripts/run_overnight.sh                          # writes experiments/ + a timestamped log
OUT_DIR=exp_night1 MODELS="Qwen/Qwen2.5-7B ibm-granite/granite-4.0-h-tiny-base" \
    bash scripts/run_overnight.sh                      # subset / custom output dir

# aggregate any set of per-case JSONs already in a dir (no model load, seconds):
python scripts/analyze_stats.py --out-dir experiments --b 20000
```

`analyze_stats.py` prints a per-(model, form) table — effect, **95% CI**, one-sided *p*, and a
`***`/`n.s.` flag (significant ⇔ CI excludes 0) — plus a significance summary and a **forest plot**
(`stats_forest.png`, colored by axis). It covers four claims: **sufficiency** (subspace−random
shift), **matched-source interchange** (subspace−norm-matched-random), **necessity** (structured-null−helix
accuracy drop at the sharing layer; `--null shuf_fourier|cov_matched|random`), and **necessity_peak**
— the same structured-null test but at the *necessity peak* depth, ingested from
`run_ablation_sweep`'s held-out structured-null pass. Empty panels (e.g. `necessity_peak` on older
JSONs) are dropped automatically. The sweep also prints Δ-vs-random and Δ-vs-shuffled-Fourier side by
side in its own peak table.

### The three metrics (validated on synthetic ground-truth cases)
- **`subspace_cos`** — principal-angle cosine between the two helix subspaces. **The primary,
  transport-relevant metric**: high (≫ floor) ⇒ the forms occupy overlapping directions, which makes
  direct transport *geometrically plausible* (tests **H1**). It does **not** guarantee transport —
  forms could share a subspace but differ in rotation/scale/offset within it, or be decoded
  differently downstream; that's what the H3 causal test settles.
- **`procrustes_cv`** — held-out R² of the best rotation aligning the two helices.
  *Necessary-not-sufficient*: high for essentially any two competent number encoders; it only
  collapses when a form has no number geometry at all (or tokenization destroyed it).
- **`linear_CKA`** — representational-similarity sanity check; weak discriminator here.

### Reading the result — this is the whole point
| `subspace_cos` | `procrustes_cv` | conclusion |
|---|---|---|
| ≫ floor | high | overlapping directions → transport is *geometrically plausible* → test it causally (step 3) |
| ~floor | high | same shape, **different directions** → transport would need a learned align-map (weaker) |
| ~floor | low | **no shared geometry** for that form → check tokenization confound, else publishable negative |

Script-axis forms aligning more than language-axis forms is evidence for **graded invariance (H2)** —
not, by itself, for a "value-not-token" mechanism (see the H2 note above).

## Results so far

Status by hypothesis. Runs use the **bug-fixed** helix code (consistent `nmax` normalization; rank-8
basis). Causal legs: **9 base models, 7 orgs, 3 non-transformer architectures** — Qwen2.5-7B,
Qwen3-8B, Qwen3-14B, Mistral-Nemo-Base, OLMo-3-7B, EuroLLM-9B, Granite-4-h-tiny (Mamba-2/MoE),
Falcon-H1-7B (parallel hybrid), Nemotron-Nano-12B (Mamba-2 hybrid); Aya-23-8B is instruct →
representational. All causal results carry **bootstrap 95% CIs + paired significance** (`analyze_stats.py`).

| leg | result | status |
|---|---|---|
| **H1/H2** graded geometry | `language` consistently least-shared; script/notation high (order varies) | ✅ all models |
| **exposure-dependent** script sharing | cross-script sharing 0.51→0.83, tracks (multi)script training | ✅ (Aya a caveat) |
| **architecture-independence** | helix + transport in **3 Mamba/hybrid** models | ✅ Granite-4, Falcon-H1, Nemotron |
| **mechanistic** localization | sharing peaks in a band, then collapses; peak layer family-specific | ✅ all |
| **H3** causal *sufficiency* | subspace patch steers all forms, **norm-matched** random does not | ✅ **45/45 cells sig** (95% CI) |
| **H3** matched-source interchange | real activation, subspace-only ≫ norm-matched random | ✅ **45/45 cells sig** |
| **H3** causal *necessity* | ablation drops accuracy helix-specifically | ◐ **script-biased, shallower than sharing**; strong vs random null (~80% script cells), weaker vs structured |

> Note on models: the causal arithmetic readout needs a **base** model (instruct Aya scores
> `clean_acc` ≈ 0 — a readout limitation, not a negative). Granite-4 & Falcon-H1 (native-`transformers`
> hybrids) run without `mamba-ssm` via the pure-PyTorch fallback; only Nemotron (remote code) requires it.

### Evidence status (claim-by-claim — the scoped view)

| claim | status |
|---|---|
| Fourier number geometry appears across the tested forms | **supported** (9 models, incl. 3 Mamba/hybrid) |
| Helix subspaces align above an isotropic random floor | **supported** |
| `language` is the least-shared axis (clean word-to-word contrasts) | **replicated** (all models) |
| `script ≈ notation` (relative order) | **model-dependent** (Qwen3 script>notation; OLMo notation>script) |
| Cross-**script** sharing tracks (multi)script training exposure | **supported, tentative** (0.51→0.83; Aya a caveat) |
| Helix + transport in **non-transformers** (Mamba/hybrid) | **supported** (Granite-4, Falcon-H1, Nemotron) |
| Cross-form helix intervention steers restricted digit-choice logits | **supported, universal** (45/45 cells, 95% CI excludes 0) |
| Steering survives **isotropic + norm-matched + matched-source** controls | **supported** (interchange 45/45); covariance/sensitivity-matched **done** via structured nulls |
| The shared subspace is *naturally necessary* | **script-biased & depth-dependent**: strong vs random null (~80% interpretable script cells), partly redundant vs structured null; language largely undefined (floor clean-acc) |
| Value is *read* earlier than it is shared | **suggestive**: necessity peaks shallower (L1–L7) than sharing (L14–L24), but *no* formal read-layer claim |
| Geometry explains behavioral numeracy gaps | **model-dependent, weak** (r 0.06→0.96 across models; frequency-confounded) |

The rest of this section elaborates each row; all headings/claims below are scoped to match it.

### H2 (graded invariance) across models — with clean contrasts
Per-axis `subspace_cos` using the **correct reference per axis** (script: en_digit↔digit-scripts;
notation: en_digit↔en_word; language: **en_word↔foreign words**, not en_digit↔words — see
`run_structure.py`; this fixes a reference-form confound that inflated the apparent language drop).

| model | org | arch | langs | script | notation | language |
|---|---|---|---|---|---|---|
| Qwen3-8B | Alibaba | transformer | 119 | **0.83** | 0.71 | 0.54 |
| Mistral-Nemo-Base | Mistral | transformer | multi | 0.80 | 0.83 | 0.60 |
| Qwen2.5-7B | Alibaba | transformer | multi | 0.77 | 0.74 | 0.52 |
| Granite-4-h-tiny | IBM | **Mamba/MoE** | 12 | 0.76 | 0.85 | 0.64 |
| Aya-23-8B | Cohere | transformer | 23 | 0.53 | 0.52 | 0.32 |
| OLMo-3-7B | AI2 | transformer | EN-primary | 0.51 | 0.76 | 0.37 |

(floor ~0.04–0.06 everywhere.) Two robust reads and one honest caveat:

- **`language` is the least-shared axis in every model** — the universal part of H2.
- **`script` sharing spans 0.51→0.83 and tracks (multi)script training exposure:** the four
  multilingual models cluster at 0.76–0.83, English-primary OLMo-3 is lowest (0.51). This is the
  freshest thread — cross-script number geometry *emerges with multiscript exposure*.
- **Caveat:** Aya (multilingual but 0.53) breaks a clean "multilingual → high," so it's more likely
  specific *numeral-script* exposure (Devanagari/Arabic-Indic) than general multilinguality — a
  hypothesis, not a law. And `script` vs `notation` order is **model-dependent** (Qwen3
  script>notation; OLMo/Granite notation>script), so we do *not* claim `script ≈ notation`.

**Architecture-independence.** Granite-4 is a **hybrid Mamba-2/MoE**, not a transformer, yet shows
the helix, the same graded sharing, causal transport, and (its cleanest-yet) cross-script necessity —
evidence the number geometry is not a transformer artifact.

### Mechanistic: sharing is localized, but the band is family-specific
The layer sweep (`run_layer_sweep.py`) shows cross-form `subspace_cos` rise, plateau, then
collapse in the final layers — *consistent with* later form-specific specialization (though the
sweep doesn't identify the cause of the decline; it could also reflect helix-fit-quality or
anisotropy changes). H2 holds at *every* layer. **Where it peaks moves with the model:**

| model | sharing peak (max `subspace_cos`) | profile |
|---|---|---|
| Qwen2.5-7B | ~L8 / 28 (mid; broad plateau L7–16) | single mid hump |
| Mistral-Nemo-Base | ~L22 / 40 (mid-late) | broad plateau L18–28 |
| Aya-23-8B | ~L19 / 32 (late) | bimodal, mid dip, late peak |

(Causal interventions are run at each model's *max-helix-R²* layer — Qwen L14, Mistral L22, Aya L25 —
which sits inside this sharing band; the sharing-`subspace_cos` peak above can differ by a few layers.)

So "shared **mid**-band" is *not* universal. What's universal: the ordering, sharing far above
floor, and the late-layer collapse. Localization becoming a finding in itself (multilingual-
specialized Aya integrates number-form later) is an honest cross-architecture result.

### H3 (sufficiency): cross-form causal transport works (single layer)
`run_transport.py` patches a source number's residual (at the sharing-peak layer) with the
`en_digit` helix's encoding of a *different* value, inside `"a + b = "`, and measures whether the
answer moves toward the transported value. On Qwen2.5-7B @ L14 (overnight run, 120 cases/form),
subspace `mean_shift` vs the random control:

| source form | subspace_shift | random_shift | ratio |
|---|---|---|---|
| en_digit (within-form) | +0.89 | +0.02 | ~44× |
| es_word (cross-language) | +1.14 | +0.00 | ~∞ |
| fr_word (cross-language) | +1.21 | +0.02 | ~60× |
| devanagari (cross-script) | +1.71 | +0.01 | ~170× |
| arabic_indic (cross-script) | +2.52 | −0.01 | ~∞ |

Patching the `en_digit` helix subspace steers arithmetic for numbers presented as Spanish/French
words and Devanagari digits — **the shared subspace is sufficient to drive the answer regardless of
surface form** — while an equal-dimension random subspace does essentially nothing. Across the
9-model overnight run this is **universal: 45/45 model×form cells significant** (bootstrap 95% CI
excludes 0), on transformers *and* the three Mamba/hybrid models. It holds against a **norm-matched**
random control (`run_necessity.py` matched-source interchange, also **45/45**: real activation,
subspace-only ≫ norm-matched random) — so the effect is the *specific helix directions*, not just "a
large enough perturbation." Effect *magnitudes* are not cross-model comparable (different readout
scales, e.g. OLMo ~5 logits vs Falcon ~0.1); significance is the claim.

> Scope: this passes **isotropic**, **norm-matched**, and **matched-source** (real-activation
> interchange) controls, and the necessity legs add covariance-matched + shuffled-Fourier structured
> nulls. The residual Makelov concern (a selected subspace acting through a parallel pathway) is
> addressed by the necessity evidence below rather than fully settled by sufficiency alone.

**Caveat — across-layer causal localization is deliberately *not* a headline.** Raw transport
magnitude isn't comparable across layers (an earlier intervention propagates through more layers →
bigger logit shift regardless of sharing). The layer-normalized `subspace/full` metric
(`run_transport_sweep.py`) fixes this for language forms but breaks for byte-fragmented scripts
(full-patch isn't a clean ceiling there). So the **single-layer** transport is the core H3 claim;
the across-layer version is at best a language-forms supplement / future work (proper causal
tracing).

### H3 (necessity): reliance is real but weaker than sufficiency — script-biased and shallower
Sufficiency shows the circuit *can* read an injected direction; necessity asks whether the model
*relies* on the shared subspace. `run_necessity.py` mean-ablates the `en_digit`-fit helix subspace
and measures the (restricted digit-choice) accuracy drop vs three nulls — Haar-random,
**covariance-matched** (random subspace in the top activation PCA), and **shuffled-Fourier** (a helix
fit through the same pipeline on shuffled labels). `run_ablation_sweep.py` repeats the random-null
ablation at every depth (discovery/test split → held-out Δ + bootstrap CI), and now also evaluates
the **structured** nulls at the discovery-selected necessity peak. Across the **9-model** overnight
run (bootstrap 95% CIs, `analyze_stats.py`):

**1. Multi-token forms must be ablated over the *whole span*, not the last token.** Number-words are
1.5–2 tokens; ablating only the last fragment leaves most of the value intact (last-token language
"necessity" is indistinguishable from the matched nulls; whole-span makes it measurable). All causal
legs use `--intervention-pos span`. Ablating the token *after* the number does ~nothing (clean
negative control).

**2. The null you pick sets the strength of the claim.** Necessity is strong against a *random*
subspace and much weaker against a *matched structured* one:

| instrument (held-out, 95% CI) | null | script cells sig | language cells sig |
|---|---|---|---|
| **layer sweep** (`run_ablation_sweep`) | Haar-random | **~80%** (21/26, clean-acc ≥ 0.6) | ~0% |
| **single layer @ sharing peak** (`run_necessity`) | shuffled-Fourier | ~50% | ~0% |

So the model relies on *structured low-rank number geometry* robustly (for scripts), but the specific
`en_digit`-helix directions are **partly redundant** with other structured directions — the honest,
reviewer-proof framing. The `--null-seeds` structured-null-at-peak pass (new) reports helix vs
shuffled-Fourier **at the necessity peak** so this claim can be stated at its strongest depth, not
just at the sharing layer.

**3. Necessity peaks *shallower* than sharing, and is script-biased.** Script necessity peaks at
L1–L7 while the causal legs run at the mid-depth *sharing* peak (L14–L24) — sufficiency (injection)
and necessity (removal) bite at different layers. Foreign number-*word* necessity is largely
**undefined**: base models often can't do es/fr arithmetic to begin with (floor clean-acc, e.g.
Falcon es 0.04, EuroLLM fr 0.17), and even where they can (Mistral es/fr 0.92) the Δ is small with
CIs touching 0. This holds across transformers *and* the three Mamba/hybrid models.

The nulls do real work: covariance-matched ablation removes *more* activation energy than the helix
yet leaves arithmetic intact, so where an effect survives it is the specific helix directions, not
removed energy.

> **No temporal / "read-layer" claim.** An earlier ablation-layer analysis suggested the value is
> "read earlier than it is shared." That does not hold up: the removed-helix energy varies wildly
> across layers/forms (E@peak ranges ~4 to ~3700), so the ablation-sweep peak tracks *how much energy
> the intervention removes*, not a read operation — and held-out splits shrink the apparent peaks.
> We report only **layer-independent** necessity (whole-span, matched-null-controlled). A real
> temporal claim needs causal tracing (see [Limitations](#limitations--planned-strengthening)).

**Synthesis (the thesis).** *Partial geometric universality, heterogeneous computational use.* The
shared number subspace is **causally sufficient** to drive arithmetic across surface forms in
multiple base families (norm-matched controlled); **reliance** on it is helix-specific but
**model-dependent** — broad across scripts and languages in Qwen (whole-span), English-digit-centric
in Mistral-Nemo. Sharing is graded (`script > language`), magnitude and reliance vary by family.

## Preliminary findings (Qwen2.5-1.5B, local validation — how the pipeline was calibrated)

A full 0–99 run on the small default model validated the pipeline and produced an early,
promising pattern. **Treat as directional only** — 1.5B has a mediocre helix (R²≈0.4–0.5);
the quantitative story needs the 7B+ cluster runs.

- **Controls behave.** Shuffled-label R² collapses to ~0.08 vs real ~0.5; random-subspace
  floor is 0.064. The helix is real and the metrics are calibrated.
- **H2 ordering already visible.** Per-axis `subspace_cos`: script ≈ notation **>** language,
  with every form far above the 0.064 floor (sharing is *graded*, not present/absent).
- **Sharing lives in a mid-network band, then declines** (`run_layer_sweep.py`): cross-form
  `subspace_cos` rises through early layers, plateaus ~L5–20 (peak L12), then collapses toward the
  floor in the final ~5 layers (consistent with later specialization; cause not identified). H2
  holds at *every* layer. Site the step-3 causal transport in the mid band (~L5–13).

### Tokenizer confound — checked, and it is NOT the explanation
The number occupies very different token counts per form (≈1.9 for digits, 3.8–4.4 for
number-words; non-Latin digits shred into 4 undecodable byte tokens — see
`scripts/inspect_tokenization.py`). So `pooling='last'` was reading number-*words* off
phonetic fragments (`treinta y siete → 'iete'`, `quarante-deux → 'ux'`). We tested whether the
language-axis drop is just this artifact by re-running under three readouts:

| axis | `last` | `mean` | `prompt_last` | floor |
|---|---|---|---|---|
| script | 0.77 | 0.75 | 0.59 | 0.06 |
| notation | 0.76 | 0.70 | 0.57 | 0.06 |
| language | 0.59 | 0.44 | 0.32 | 0.06 |

The language drop **persists (and widens) under every readout**, so it is *not* a pooling
artifact — script/notation genuinely share the helix more than number-words do. Readout choice
*does* move the magnitudes, so we pin a principled primary: **`mean`-over-span** (the default),
which avoids both the last-fragment confound and the different-carrier-token confound that
`prompt_last` introduces for es/fr (read at `es`/`est`, not `is`). `last` and `prompt_last` are
reported as robustness checks.

## Known threats (design around these, don't discover them in review)
1. **Interpretability illusion** ([Makelov et al., 2311.17030](https://arxiv.org/abs/2311.17030)):
   a subspace patch can change behavior via a *dormant parallel pathway* even if it isn't the
   model's real mechanism. The step-3 controls in `src/patching.py` (isotropic + norm-matched +
   covariance-matched + shuffled-Fourier) *mitigate* this — but isotropic controls alone don't settle
   it; covariance/sensitivity-matched *interchange* controls are still pending. Necessity helps.
2. **Tokenization confounds** — *checked (see Preliminary findings)*: `37` vs `thirty-seven` vs
   `३७` follow different token paths and BPE shreds multi-digit strings. Verified the H2 result
   is robust to the readout via `--pooling {last,mean,prompt_last}`; `mean`-over-span is primary.
   Re-verify on each new model, since tokenization differs.
3. **Decodability ≠ causal use**: a probe finding the helix ≠ the model using it. The causal
   transport step plus controls are what close this gap.

## Limitations & planned strengthening

External-review weaknesses and their status. ✅ = addressed; ◐ = partly; ☐ = open.

- ✅ **"Read-layer" temporal claim — DROPPED.** The ablation-layer Δ is confounded (earlier
  interventions propagate further; removed energy varies wildly with depth, ~4 to ~3700; winner's
  curse). We now report only layer-independent, whole-span necessity, and use a discovery/test split.
  Any temporal claim would need causal tracing / path patching — ☐ open, not attempted.
- ✅ **H2 reference-form confound — FIXED.** Clean contrasts use the correct reference per axis
  (language = en_word↔foreign words), and the full pairwise matrix + word-to-word cells are reported
  (`run_structure.py`). The ordering survives; the language gap shrinks.
- ✅ **Final-token-only interventions — FIXED.** `--intervention-pos {last,span,after}`; whole-span
  ablation is required for multi-token forms (last-token under-ablates). Token counts are recorded.
  *(At 9-model scale the earlier "language necessity is real" read did not hold — necessity is
  script-biased and often undefined for foreign words at floor clean-acc; see H3 necessity.)*
- ✅ **Matched controls — ADDED.** Ablation is tested against covariance-matched and shuffled-Fourier
  nulls (not just Haar-random), with removed-energy reported; interchange uses a norm-matched random
  control; per-seed curves are kept (no rounding) with mean±std.
- ◐ **"Restricted digit-choice accuracy," not arithmetic accuracy.** Renamed throughout; the readout
  is still argmax over the ten 0–9 tokens on single-digit sums. ☐ *Open:* extend to multi-digit,
  subtraction, comparison, full-continuation likelihood, word-form outputs, and same-representation
  operand+answer (main-conference reach).
- ✅ **Reproducibility.** Compact result JSONs (model revision hash, intervention norms, per-seed
  curves) are now committed (`.gitignore` keeps only large figures/caches out).
- ◐ **Statistical power.** The causal legs now log **per-case** outcomes (`per_case_shift` in
  transport; `per_case` clean/helix/controls in necessity + interchange; `per_case_heldout_peak` in
  the ablation sweep). `scripts/analyze_stats.py` reads those and reports **bootstrap 95% CIs +
  paired significance** per (model, form): sufficiency (subspace−random shift), necessity
  (structured-null−helix accuracy drop, vs shuffled-Fourier by default), and matched-source
  interchange (subspace−norm-matched-random). A finding is significant only when its 95% CI
  **excludes 0**. `analyze_stats.py` now also reports a **sign-flip permutation test** (a proper
  null-centered p, not just the bootstrap), a **Benjamini-Hochberg FDR** correction across cells per
  claim, and a **per-model aggregation** (so "45 individually significant cells" is not the unit of
  analysis) — addressing audit #11. ☐ *Open:* raise cases/form and cluster-bootstrap by source value
  (needs per-case key logging) for fully independence-aware CIs.

### Code audit (2026-07) — status
A external code+measurement audit (`experiments/crosslingual_number_helix_code_audit.md`) is largely
addressed. **Numerical/reproducibility (require regenerating magnitudes):**
- ✅ **Fourier basis centering (#3).** The fit regressed centered PCA scores on an *uncentered* basis
  with no intercept, corrupting R² and every reconstructed vector (R² on a clean linear code 0.66→**1.0**).
  Now centered; `B_mean` stored and reused in reconstruction. **⚠ invalidates prior reconstruction-based
  transport magnitudes and R² — rerun before quoting numbers.**
- ✅ **Deterministic PCA (#8)** (`svd_solver="full"`) in fit, cov-matched control, Procrustes; ✅ **Procrustes
  leakage** — PCA/centering now fit on the train split only.
- ✅ **Hook-equivalence test (#4).** `verify_hook_layer` asserts `hidden_states[L]==block L-1 output` per
  architecture (prints rel-error at startup; 0.0 on Qwen) — the assumption behind the cross-architecture claim.
- ✅ **Whole-span energy (#5/6)** (was last-token, first-seed only); ✅ **shared case set + honest denominator (#10)**.

**Measurement scope (new instruments):**
- ✅ **Coordinate-level identity (#1).** `canonical_map_cosines` reports per-feature signed cosine (not just
  span overlap); README wording split into "shared subspace" vs "shared coordinates".
- ✅ **Matched-arithmetic delta transport (#2)** — `run_transport.py --delta` transports only
  `QQᵀ(h_en(a′,b)−h_en(a,b))` (holds addend/syntax/format/offset fixed); a `delta_transport` claim in `analyze_stats`.
- ✅ **Held-out helix R² + pipeline-matched permutation null (#7)**; ✅ **context-matched ablation baseline (#12)**
  (`--ablation-baseline form_arith`, now default); ✅ **answer-token validation (#9)** (`validate_single_token_answers`).
- ✅ **Test suite** (`tests/test_core.py`, **22 tests**) locking in the above.

**Round 2 (post-overhaul audit) — the "before any rerun" gate, all addressed:**
- ✅ **FDR now uses `perm_p` (#1).** BH-FDR was applied to the bootstrap tail (not a null-centered p);
  it now corrects the **sign-flip permutation** p. The bootstrap field is renamed `boot_tail`.
- ✅ **Fail-fast answer tokens (#2/#9).** `continuation_answer_ids` derives each id from the real
  continuation after `"a + b = "`, asserts one token that decodes to the digit, and **raises** (no
  last-sub-token fallback). Wired into all five causal/structure scripts.
- ✅ **Multi-seed, norm-matched delta controls (#3).** `--delta-ctrl-seeds` (default 10); each control
  delta is rescaled to `‖QQᵀΔ‖` and averaged, so a bigger effect can't just be a bigger perturbation.
- ✅ **Hook equivalence enforced everywhere (#4).** `assert_hook_equivalence` (fail-fast) runs at
  startup in transport / necessity / sweep and the rel-error is saved to each JSON (`hook_rel_error`).
- ✅ **Exact case pairing + clustered inference (#11/#12).** `paired()` asserts equal length (never
  truncates); transport logs `per_case_keys`; `analyze_stats` adds a **cluster bootstrap by source
  value** (wider, independence-aware CI) and uses it for the significance flag when keys are present.
- ✅ **Rank-aware overlap (#6)** (`subspace_overlap`: shared energy + rank-penalized); ✅ **permutation
  nulls default 500 + add-one p-value (#7)**; ✅ **clean-contrast H2 aggregation (#8)** (`aggregate_runs`
  now defaults to `structure_*.json` clean contrasts and warns on the confounded `align_*.json` path);
  ✅ **`top_pca_span_basis` rename (#14)** (alias kept); ✅ **"read-layer" language removed from the
  sweep (#15)** → "ablation-sensitivity peak".
- ☐ **Open (need new data / decisions):** carrier-language factorial (#16); position/token-count-conditioned
  ablation baselines + post-span & joint-span interventions (#9/#10); preregistered causal-layer selection;
  full continuation-likelihood / word-form output readouts; coordinate-level bootstrap CIs + cross-prediction (#5).
- ☐ **Final concurrent-work search before submission.** Related work verified 2026-07-18 (Gupta
  blog; Lan/Torr/Barez 2311.04131; FARS 2605.09496; Semantic Hub 2411.04986) — the novelty is scoped
  accordingly. Re-search close to submission for concurrent cross-form / same-coordinate number work,
  since this space moves monthly.

## Roadmap
- [x] Step 1 — reproduce the helix fit per form
- [x] Step 2 — cross-form subspace alignment + Procrustes-CV + CKA (the go/no-go signal)
- [x] Local validation on Qwen2.5-1.5B: controls calibrated, H2 ordering visible
- [x] Tokenizer-confound check: language drop robust across `last`/`mean`/`prompt_last`; primary readout pinned to `mean`
- [x] Tooling: per-layer sweep, cross-run aggregator, tokenization diagnostic
- [x] **Real run — Qwen2.5-7B**: H2 (graded invariance) replicated, sharing localized to a mid band
- [x] **Step 3 sufficiency — causal transport** + **norm-matched** control (Qwen + Mistral-Nemo)
- [x] **Step 3 necessity — whole-span ablation vs matched nulls** (cov-matched + shuffled-Fourier): model-dependent (Qwen broad, Mistral English-digits)
- [x] **#6 pairwise matrix + clean H2 contrasts** (reference-form confound fixed); #7 geometry↔behavior model-dependent (dropped as headline)
- [x] **External-review hardening**: bug fixes; matched controls; multi-token interventions; held-out splits; committed result JSONs; **read-layer claim dropped**
- [x] **Family expansion → 9 base causal models / 7 orgs / 3 non-transformers**: + Qwen3-14B, EuroLLM-9B, Falcon-H1-7B (parallel hybrid), Nemotron-Nano-12B (Mamba-2 hybrid) → the **exposure-dependent script-sharing** + **architecture-independence** threads
- [x] **Statistics infrastructure**: per-case logging in all causal legs + `analyze_stats.py` (bootstrap 95% CIs, paired significance, forest plot) + `run_overnight.sh` (unattended full-model loop, per-model cache cleanup)
- [x] **9-model overnight run**: sufficiency **45/45** + interchange **45/45** significant; necessity script-biased & shallower than sharing (see H3 necessity)
- [x] **Per-layer structured nulls**: `run_ablation_sweep` now tests helix vs cov-matched + shuffled-Fourier **at the necessity peak** (held-out CIs) → `necessity_peak` claim in `analyze_stats`
- [ ] **Re-run the overnight loop** to populate `necessity_peak` (structured-null-at-peak) across all 9 models; then finalize the necessity claim at its strongest depth
- [ ] **Gemma-4** (multimodal-loader verify) + **Universality — Llama-3.1-8B (base)** (blocked on HF gated-repo approval; the original helix model)
- [ ] **Extend eval** (main-conf reach): multi-digit, subtraction, comparison, word-form outputs, same-representation operand+answer; more cases/form for tighter CIs
- [ ] Optional temporal claim: causal tracing / path patching (only if pursuing the representation-vs-use question)
- [ ] Time arm — dates/years (DateAugBench has format-invariance puzzles, 2505.16088)
- [ ] Write-up — figures: pairwise heatmap, whole-span necessity (matched nulls), transport (norm-matched), layer sweep; vs FARS (2605.09496) + universal-numbers (2510.26285)

## Key references
- Kantamneni & Tegmark, *LLMs Use Trigonometry to Do Addition* — [2502.00873](https://arxiv.org/abs/2502.00873) ([code](https://github.com/subhashk01/LLM-addition))
- Engels et al., *Not All Features Are One-Dimensionally Linear* — [2405.14860](https://arxiv.org/abs/2405.14860)
- Gurnee & Tegmark, *LMs Represent Space and Time* — [2310.02207](https://arxiv.org/abs/2310.02207)
- *Separating Tongue from Thought* (cross-lingual patching of language-agnostic **lexical** concepts — separable concept/language info; not numerical-value Fourier geometry) — [2411.08745](https://arxiv.org/abs/2411.08745)
- *Effect of Scripts and Formats on LLM Numeracy* (behavioral only) — [2601.15251](https://arxiv.org/abs/2601.15251)
- *Language Models Learn Universal Representations of Numbers* (universal across **models**, not forms) — [2510.26285](https://arxiv.org/abs/2510.26285)
- **FARS — Format-Agnostic Reasoning Subspaces** (closest *method*: PCA-subspace + patching + ablation + cross-arch; reasoning concepts, not the helix/numbers) — [2605.09496](https://arxiv.org/abs/2605.09496)
- **Gupta**, *From Latin Digits to Babylonian Cuneiform: Number Helices Across Scripts* (June 2026 blog; cross-script helix *shape*, descriptive, no causal) — [girishgupta.com](https://girishgupta.com/beyond-the-parrot/20260618-from-latin-digits-to-babylonian-cuneiform)
- **Lan, Torr & Barez**, *Towards Interpretable Sequence Continuation: Analyzing Shared Circuits* (EMNLP 2024; shared **circuits** across numerals/words/months, EN/ES, via ablation) — [2311.04131](https://arxiv.org/abs/2311.04131)
- *The Semantic Hub Hypothesis* (digit/word shared English-anchored space + steering) — [2411.04986](https://arxiv.org/abs/2411.04986)
- Makelov et al., *An Interpretability Illusion for Subspace Activation Patching* — [2311.17030](https://arxiv.org/abs/2311.17030)
