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

Evidence, on Qwen2.5-7B (base) and replicated representationally on Aya-23-8B:

- **Graded sharing (H2):** cross-form alignment falls `script ≈ notation > language`, every form far
  above a random floor — robust across two model families and three activation readouts.
- **Localized, family-specifically:** sharing peaks in a layer band then collapses; the band is
  mid-network in Qwen, late in Aya (the ordering, not the location, is what's universal).
- **Causally sufficient everywhere:** patching the shared subspace with a value steers arithmetic
  for *every* form (Spanish/French/German words, Devanagari/Arabic-Indic digits), while a random
  subspace does nothing — the illusion control passes.
- **Only partially necessary:** ablating the shared subspace hurts most forms *some*, but not
  uniformly — English forms most, other scripts/languages less — so most forms retain redundant
  value encoding outside it. Necessity is graded and moderately tracks representational sharing
  (r≈0.55, n=7); it does **not** split cleanly by script vs language (n≈24/form, noisy).

**Net:** the shared subspace is *universally sufficient* to drive arithmetic across forms and only
*partially necessary* (forms keep independent value encoding to varying degrees). Sharing is graded
— cleanly `script > language` **representationally** — but the causal necessity gradient is noisier
and not a clean script/language dichotomy.

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
> **H2 (value- not token-driven).** Forms that differ only in *script* (`37` vs `३७`) share the
> geometry more than forms that differ in *language* (`thirty-seven` vs `treinta y siete`),
> because sharing tracks the underlying value rather than the tokens.
>
> **H3 (causal transport).** A number written in form B can be activation-patched onto the
> form-A helix and shift the model's downstream arithmetic toward that value — using only the
> A-helix subspace, surviving the controls below.

Any clear outcome is a paper. **Even a clean negative** — "the helix is *not* shared across
forms, and here's the tokenization-controlled evidence" — is publishable if pre-registered,
because it contradicts the implicit universality assumption in the number-geometry literature.

## What's novel (the gap, verified against the literature)

- Number-geometry papers are **monolingual English digits** — no languages, scripts, or words
  ([2502.00873](https://arxiv.org/abs/2502.00873), [2510.26285](https://arxiv.org/abs/2510.26285)).
- The cross-lingual concept-patching method exists but was applied **only to concrete nouns,
  never numbers** ([Separating Tongue from Thought, 2411.08745](https://arxiv.org/abs/2411.08745)).
- Scripts/formats numeracy work is **behavioral only** — it measures accuracy, never the
  internal geometry ([2601.15251](https://arxiv.org/abs/2601.15251),
  [2505.16088](https://arxiv.org/abs/2505.16088)).

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
scripts/run_necessity.py      # STEP 3 necessity: helix-subspace ablation + matched-source interchange
scripts/run_transport_sweep.py   # transport at every layer (layer-normalized subspace/full)
scripts/aggregate_runs.py     # collect experiments/align_*.json -> cross-model table + bar chart
scripts/inspect_tokenization.py  # diagnostic: token counts + what each pooling reads per form
```

**Three axes of variation** (`src/data.py`), ordered by how directly they test a
*value-driven* shared helix. All forms render the *same integer set in the same order*, so
activation rows are paired across forms (this is what makes the comparisons valid):
- **Script axis — HEADLINE** (same language + notation, only glyphs change):
  `en_digit` `37` vs `devanagari_digit` `३७` vs `arabic_indic_digit` `٣٧` vs `fullwidth_digit` `３７`.
  A shared helix here is near-pure evidence of value-driven geometry, and it's the
  least-covered contribution vs prior work (FARS 2605.09496 used Latin-script prose only).
- **Notation axis** (digits vs spelled-out words, language fixed): `en_digit` vs `en_word`.
- **Language axis** (spelled-out words, language varies): `en_word`, `es_word`, `fr_word`, `de_word`.

**H2 prediction:** if sharing is value-driven, per-axis alignment should fall
`script ≥ notation ≥ language`. `run_fit_and_align.py` prints this per-axis summary directly.

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

### The three metrics (validated on synthetic ground-truth cases)
- **`subspace_cos`** — principal-angle cosine between the two helix subspaces. **The primary,
  transport-relevant metric**: high (≫ floor) ⇒ same *literal directions* ⇒ a direct activation
  patch can carry a number from one form to the other (tests **H1**).
- **`procrustes_cv`** — held-out R² of the best rotation aligning the two helices.
  *Necessary-not-sufficient*: high for essentially any two competent number encoders; it only
  collapses when a form has no number geometry at all (or tokenization destroyed it).
- **`linear_CKA`** — representational-similarity sanity check; weak discriminator here.

### Reading the result — this is the whole point
| `subspace_cos` | `procrustes_cv` | conclusion |
|---|---|---|
| ≫ floor | high | **same directions** → H1 holds → direct patch works → build step 3 |
| ~floor | high | same shape, **different directions** → transport needs a learned align-map (weaker positive) |
| ~floor | low | **no shared geometry** for that form → check tokenization confound, else publishable negative |

If **script-axis forms align more than language-axis forms**, that's direct evidence for **H2**.

## Results so far

Status by hypothesis (as of this writing; Llama-3.1-8B pending HF gated-repo approval).
All results use the **bug-fixed** helix code (consistent `nmax` normalization; rank-8 basis, see
below). The fixes left the representational results essentially unchanged.

| leg | result | status |
|---|---|---|
| **H1/H2** shared, graded geometry | `script ≈ notation > language`, all ≫ floor | ✅ Qwen2.5-7B **and** Aya-23-8B |
| **mechanistic** localization | sharing peaks in a band, then collapses late | ✅ but band is **family-specific** |
| **H3** causal *sufficiency* | subspace patch steers all forms, random does not | ✅ Qwen2.5-7B (base) |
| **H3** causal *necessity* | ablation breaks scripts, not languages (graded) | ✅ Qwen2.5-7B (base) |

> Note on models: the causal arithmetic readout needs a **base** model. Qwen2.5-7B and
> Llama-3.1-8B are base; Aya-23-8B is instruction-tuned, so it is used for the *representational*
> results only (its causal `clean_acc` ≈ 0 — a readout limitation, not a negative result).

### H2 confirmed at scale, and replicated across families
Per-axis `subspace_cos` (mean-pooled, vs `en_digit`), on real models:

| axis | Qwen2.5-7B | Aya-23-8B | floor |
|---|---|---|---|
| script | 0.70 | 0.53 | ~0.04 |
| notation | 0.69 | 0.51 | ~0.04 |
| language | 0.43 | 0.32 | ~0.04 |

`script ≈ notation > language` holds in **two independent families**, every form far above the
floor. Note the magnitude is *not* universal — Aya (built for multilinguality) shows **weaker**
sharing than Qwen, so what's robust is the **ordering**, not the amount.

### Mechanistic: sharing is localized, but the band is family-specific
The layer sweep (`run_layer_sweep.py`) shows cross-form `subspace_cos` rise, plateau, then
collapse in the final layers — a shared-value → form-specific-output arc, with H2 holding at
*every* layer. **But where it peaks moves with the model:**

| model | sharing peak | profile |
|---|---|---|
| Qwen2.5-7B | ~L14 / 28 (mid) | single mid hump |
| Aya-23-8B | ~L25 / 32 (late) | bimodal, mid dip, late global peak |

So "shared **mid**-band" is *not* universal. What's universal: the ordering, sharing far above
floor, and the late-layer collapse. Localization becoming a finding in itself (multilingual-
specialized Aya integrates number-form later) is an honest cross-architecture result.

### H3 (sufficiency): cross-form causal transport works (single layer)
`run_transport.py` patches a source number's residual (at the sharing-peak layer) with the
`en_digit` helix's encoding of a *different* value, inside `"a + b = "`, and measures whether the
answer moves toward the transported value. On Qwen2.5-7B @ L14, subspace `mean_shift` vs the
random control:

| source form | subspace_shift | random_shift | ratio |
|---|---|---|---|
| en_digit (within-form) | +0.98 | +0.03 | ~39× |
| es_word (cross-language) | +1.13 | −0.00 | ~∞ |
| fr_word (cross-language) | +1.11 | +0.04 | ~31× |
| devanagari (cross-script) | +1.79 | +0.01 | ~224× |

Patching the `en_digit` helix subspace steers arithmetic for numbers presented as Spanish/French
words and Devanagari digits — **the model reads the shared helix regardless of surface form** —
while an equal-dimension *random* subspace does essentially nothing (the interpretability-illusion
control passes). Transport strength tracks the step-2 ordering (script > language).

**Caveat — across-layer causal localization is deliberately *not* a headline.** Raw transport
magnitude isn't comparable across layers (an earlier intervention propagates through more layers →
bigger logit shift regardless of sharing). The layer-normalized `subspace/full` metric
(`run_transport_sweep.py`) fixes this for language forms but breaks for byte-fragmented scripts
(full-patch isn't a clean ceiling there). So the **single-layer** transport is the core H3 claim;
the across-layer version is at best a language-forms supplement / future work (proper causal
tracing).

### H3 (necessity): the model *naturally uses* the shared subspace — and it's graded
Sufficiency shows the circuit *can* read an injected direction; necessity asks whether the model
*relies* on the shared subspace when processing a foreign-form number. `run_necessity.py` runs two
tests on Qwen2.5-7B @ L14 (multi-seed random controls):

**(A) Ablation** — mean-ablate the `en_digit`-fit helix subspace from a *source* number and re-measure arithmetic accuracy:

| form | clean | helix-ablate | random-ablate | Δ (rand − helix) |
|---|---|---|---|---|
| en_digit | 0.83 | 0.21 | 0.83 ± 0.05 | **0.62** |
| devanagari (script) | 0.92 | 0.54 | 0.93 ± 0.02 | **0.39** |
| es_word (language) | 0.62 | 0.54 | 0.60 ± 0.02 | 0.06 |
| fr_word (language) | 0.75 | 0.67 | 0.73 ± 0.02 | 0.06 |

Ablating the shared subspace **breaks cross-script arithmetic** but **barely dents number-words** —
so languages keep value information *outside* the shared subspace. Random-subspace ablation does
nothing anywhere.

**(B) Matched-source interchange** — transport the model's *real* `en_digit` activation (not the
Fourier reconstruction), subspace-only: `subspace_shift` ≫ `random_shift` for **all** forms
(es 1.04, fr 1.00, devanagari 1.61, en 0.92; random ≈ 0). So sufficiency holds with genuine
activations, independent of fit quality.

**Synthesis (the thesis).** Sufficient everywhere, necessary only for scripts →
**cross-script sharing is deep (necessary + sufficient); cross-language sharing is sufficient-only**
(number-words retain an independent value encoding). The causal necessity gradient *matches* the
representational `subspace_cos` gradient — two independent methods agreeing that sharing degrades
with surface-transformation distance.

## Preliminary findings (Qwen2.5-1.5B, local validation — how the pipeline was calibrated)

A full 0–99 run on the small default model validated the pipeline and produced an early,
promising pattern. **Treat as directional only** — 1.5B has a mediocre helix (R²≈0.4–0.5);
the quantitative story needs the 7B+ cluster runs.

- **Controls behave.** Shuffled-label R² collapses to ~0.08 vs real ~0.5; random-subspace
  floor is 0.064. The helix is real and the metrics are calibrated.
- **H2 ordering already visible.** Per-axis `subspace_cos`: script ≈ notation **>** language,
  with every form far above the 0.064 floor (sharing is *graded*, not present/absent).
- **Sharing lives in a mid-network band, then re-specializes** (`run_layer_sweep.py`): cross-form
  `subspace_cos` rises through early layers, plateaus ~L5–20 (peak L12), then collapses toward the
  floor in the final ~5 layers — a shared-value → form-specific-output arc. H2 holds at *every*
  layer (the axis curves never cross). Site the step-3 causal transport in the mid band (~L5–13).

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
   model's real mechanism. The step-3 controls in `src/patching.py` exist for exactly this — a
   successful transport is necessary but not sufficient.
2. **Tokenization confounds** — *checked (see Preliminary findings)*: `37` vs `thirty-seven` vs
   `३७` follow different token paths and BPE shreds multi-digit strings. Verified the H2 result
   is robust to the readout via `--pooling {last,mean,prompt_last}`; `mean`-over-span is primary.
   Re-verify on each new model, since tokenization differs.
3. **Decodability ≠ causal use**: a probe finding the helix ≠ the model using it. The causal
   transport step plus controls are what close this gap.

## Roadmap
- [x] Step 1 — reproduce the helix fit per form
- [x] Step 2 — cross-form subspace alignment + Procrustes-CV + CKA (the go/no-go signal)
- [x] Local validation on Qwen2.5-1.5B: controls calibrated, H2 ordering visible
- [x] Tokenizer-confound check: language drop robust across `last`/`mean`/`prompt_last`; primary readout pinned to `mean`
- [x] Tooling: per-layer sweep, cross-run aggregator, tokenization diagnostic
- [x] **Real run — Qwen2.5-7B**: H2 confirmed, sharing localized to a mid band
- [x] **Step 3 sufficiency — causal transport** (single layer) with full/subspace/random controls
- [x] **Step 3 necessity — ablation + matched-source interchange**: graded, script-necessary/language-sufficient-only
- [x] **Universality — Aya-23-8B** (representational): H2 replicates; localization is family-specific (late peak)
- [x] **Bug fixes** (external review): consistent `nmax` normalization + rank-8 basis / SVD orthonormalization; all runs redone
- [ ] **Universality — Llama-3.1-8B (base)** (blocked on HF gated-repo approval; the original helix model) — repr. + causal
- [ ] Pairwise form×form alignment matrix (word-to-word) + geometry↔behavior link (`subspace_cos` vs arithmetic accuracy)
- [ ] Hardening: norm-matched random control; stronger (covariance-matched) alignment null
- [ ] Time arm — dates/years (DateAugBench has format-invariance puzzles, 2505.16088)
- [ ] Write-up — figures assembled, related-work positioning vs FARS (2605.09496) + universal-numbers (2510.26285)

## Key references
- Kantamneni & Tegmark, *LLMs Use Trigonometry to Do Addition* — [2502.00873](https://arxiv.org/abs/2502.00873) ([code](https://github.com/subhashk01/LLM-addition))
- Engels et al., *Not All Features Are One-Dimensionally Linear* — [2405.14860](https://arxiv.org/abs/2405.14860)
- Gurnee & Tegmark, *LMs Represent Space and Time* — [2310.02207](https://arxiv.org/abs/2310.02207)
- *Separating Tongue from Thought* (cross-lingual concept patching) — [2411.08745](https://arxiv.org/abs/2411.08745)
- *Effect of Scripts and Formats on LLM Numeracy* — [2601.15251](https://arxiv.org/abs/2601.15251)
- *Language Models Learn Universal Representations of Numbers* (universal across models, disclaims cross-lingual) — [2510.26285](https://arxiv.org/abs/2510.26285)
- *FARS — Format-Agnostic Reasoning Subspaces* (closest adjacent method; general concepts, not the helix) — [2605.09496](https://arxiv.org/abs/2605.09496)
- Makelov et al., *An Interpretability Illusion for Subspace Activation Patching* — [2311.17030](https://arxiv.org/abs/2311.17030)
