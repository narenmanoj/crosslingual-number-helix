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

Thesis: **partial geometric universality, heterogeneous computational use.** Evidence:

- **Graded sharing (H2):** with clean per-axis contrasts, cross-form alignment falls
  `script ≈ notation > language`, every form far above a random floor — robust across **three**
  model families (Qwen2.5-7B, Mistral-Nemo-Base, Aya-23-8B) and three activation readouts. Magnitude
  varies by family; the *ordering* doesn't.
- **Localized, family-specifically:** sharing peaks in a layer band then collapses; mid-network in
  Qwen, mid-late in Mistral-Nemo, late in Aya (the ordering, not the location, is universal).
- **Causally *sufficient* everywhere:** patching the shared subspace with a value steers arithmetic
  for *every* form, while a **norm-matched** random subspace does essentially nothing (Qwen2.5-7B,
  Mistral-Nemo-Base). This is the strong, universal causal claim.
- **Causally *necessary*, but model-dependent, and only under whole-span ablation:** ablating the
  shared subspace (vs covariance-matched and shuffled-Fourier nulls) *drops* arithmetic accuracy —
  helix-specifically — but for multi-token number-words you must ablate the **whole span**, not the
  last token (last-token under-ablates). Under whole-span ablation **Qwen** relies on the shared
  subspace across scripts *and* languages; **Mistral-Nemo** relies mainly for English digits and is
  otherwise redundant (its apparent cross-form reliance was winner's curse, gone on held-out data).

**Net:** number geometry is partially shared across scripts, notations, and languages; the shared
directions are **causally sufficient** to drive arithmetic in every form, while **reliance** on them
is helix-specific but **model-dependent** (broad in Qwen, English-digit-centric in Mistral-Nemo).
Sharing is graded (`script > language`). **Causal results need base models; Aya (instruct) is
representational-only.** We make *no* temporal "read-layer" claim — see
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
> **H2 (graded invariance).** Cross-form compatibility is greatest under glyph-only changes
> (`37`↔`३७`), intermediate under digit→word changes (`37`↔`thirty-seven`), and weaker under
> language changes (`thirty-seven`↔`treinta y siete`). *This states what is measured — a gradient
> of surface-form invariance — without claiming the mechanism is "value not token": script variants
> also share positional-decimal notation, digit-wise composition, and an English carrier.*
>
> **H3 (causal transport & reliance).** A number written in form B can be activation-patched onto
> the form-A helix and shift the model's downstream arithmetic toward that value (sufficiency), and
> ablating the shared subspace can disrupt arithmetic in form B (necessity) — both against matched
> controls. High subspace overlap makes transport *geometrically plausible*; the causal test is what
> shows the downstream computation actually interprets the transported coordinates.

Any clear outcome is a paper. **Even a clean negative** — "the helix is *not* shared across
forms, and here's the tokenization-controlled evidence" — is publishable if pre-registered,
because it contradicts the implicit universality assumption in the number-geometry literature.

## What's novel (narrow, and stated carefully)

Prior work already establishes a lot of the neighborhood — the novelty is a *specific combination*,
not a fundamentally new phenomenon or method:

- **Helix / Fourier number geometry** is established ([2502.00873](https://arxiv.org/abs/2502.00873),
  [2405.14860](https://arxiv.org/abs/2405.14860)); some work also *observes* similar helix spectra
  across numeral systems, and universality *across models* ([2510.26285](https://arxiv.org/abs/2510.26285)).
- **Shared circuitry between Arabic numerals and number words** (and cross-language effects) has been
  studied; **cross-format / cross-lingual activation patching** exists (FARS
  [2605.09496](https://arxiv.org/abs/2605.09496) for reasoning concepts; *Separating Tongue from
  Thought* [2411.08745](https://arxiv.org/abs/2411.08745) for concept nouns).

**What remains unresolved — our contribution:** whether these forms occupy the *same Fourier-structured
residual-stream coordinates* (literal subspace overlap), whether those coordinates support *direct
cross-form arithmetic transport* (natural-activation interchange, not just a reconstruction), whether
models *naturally rely* on them (necessity, not only sufficiency), and how this varies by architecture
— including **non-Latin numeral scripts**. That package is not trivially implied by the prior work
above.

> ⚠️ *Related-work TODO:* verify and cite the recent cross-numeral-system helix-spectra work and the
> numeral↔number-word shared-circuit work (flagged in review) before submission; the framing above is
> written to be correct regardless, but the citations must be pinned down.

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
scripts/run_ablation_sweep.py    # necessity vs layer: helix-ablation Δ per layer (exploratory; NOT a read-layer)
scripts/run_transport_sweep.py   # transport at every layer (layer-normalized subspace/full)
scripts/run_structure.py      # #6 pairwise form x form matrix + #7 geometry<->behavior (one model load)
scripts/aggregate_runs.py     # collect experiments/align_*.json -> cross-model table + bar chart
scripts/inspect_tokenization.py  # diagnostic: token counts + what each pooling reads per form
```

**Three axes of variation** (`src/data.py`), ordered by increasing surface-form distance. All forms
render the *same integer set in the same order*, so activation rows are paired across forms (this is
what makes the comparisons valid):
- **Script axis — HEADLINE** (same language + notation, only glyphs change):
  `en_digit` `37` vs `devanagari_digit` `३७` vs `arabic_indic_digit` `٣٧` vs `fullwidth_digit` `３７`.
  The least-covered contribution vs prior work (esp. non-Latin numeral scripts). *Note: high sharing
  here shows glyph-invariance; it doesn't by itself isolate "value" from shared positional-decimal
  notation.*
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

Status by hypothesis (as of this writing; Llama-3.1-8B pending HF gated-repo approval).
All results use the **bug-fixed** helix code (consistent `nmax` normalization; rank-8 basis, see
below). The fixes left the representational results essentially unchanged.

| leg | result | status |
|---|---|---|
| **H1/H2** shared, graded geometry | `script ≈ notation > language` (clean contrasts), all ≫ floor | ✅ Qwen2.5-7B, Mistral-Nemo-Base, Aya-23-8B |
| **mechanistic** localization | sharing peaks in a band, then collapses late | ✅ band is **family-specific** |
| **H3** causal *sufficiency* | subspace patch steers all forms, **norm-matched** random does not | ✅ Qwen2.5-7B + Mistral-Nemo (base) |
| **H3** causal *necessity* | whole-span ablation drops accuracy, helix-specifically | ✅ **model-dependent**: Qwen broad (scripts+languages), Mistral English-digits mainly |

> Note on models: the causal arithmetic readout needs a **base** model. Qwen2.5-7B, Mistral-Nemo-Base,
> and Llama-3.1-8B are base; Aya-23-8B is instruction-tuned, so it is used for the *representational*
> results only (its causal `clean_acc` ≈ 0 — a readout limitation, not a negative result).

### Evidence status (claim-by-claim — the scoped view)

| claim | status |
|---|---|
| Fourier number geometry appears across the tested forms | **supported** |
| Helix subspaces align above an isotropic random floor | **supported** |
| `script ≈ notation > language` (clean word-to-word contrasts) | **replicated** (3 families) |
| Cross-form helix intervention steers restricted digit-choice logits | **supported** |
| Steering survives **isotropic + norm-matched** controls | **supported**; covariance/sensitivity-matched interchange controls **pending** |
| The shared subspace is *naturally necessary* (whole-span, matched nulls) | **model/form-dependent**: Qwen broad; Mistral English-digits; some number-words still modest |
| Ablation Δ peaks earlier than alignment | **observed descriptively** (confounded — see Limitations) |
| Value is *read* earlier than it is shared | **not established** |
| Geometry explains behavioral numeracy gaps | **not supported** (weak, frequency-confounded) |

The rest of this section elaborates each row; all headings/claims below are scoped to match it.

### H2 (graded invariance) replicated across families — with clean contrasts
Per-axis `subspace_cos` using the **correct reference per axis** (script: en_digit↔digit-scripts;
notation: en_digit↔en_word; language: **en_word↔foreign words**, not en_digit↔words — see
`run_structure.py`). This fixes a reference-form confound: comparing en_digit to foreign *words*
changes both notation and language, and inflates the apparent language drop.

| axis | Qwen2.5-7B | Mistral-Nemo-Base | floor |
|---|---|---|---|
| script | 0.77 | 0.80 | ~0.04 |
| notation | 0.74 | 0.83 | ~0.04 |
| language | 0.52 | 0.60 | ~0.04 |

`script ≈ notation > language` holds in **three independent families** (Aya replicates the ordering;
see the JSONs), every form far above the floor. Two honest notes: the magnitude is *not* universal
(Aya is weaker), and with the clean contrast the **language gap is smaller** than an en_digit-referenced
table suggests — number-words form their own moderately-shared cluster (e.g. Qwen `en_word↔es_word`
0.55, `es_word↔fr_word` 0.50).

### Mechanistic: sharing is localized, but the band is family-specific
The layer sweep (`run_layer_sweep.py`) shows cross-form `subspace_cos` rise, plateau, then
collapse in the final layers — *consistent with* later form-specific specialization (though the
sweep doesn't identify the cause of the decline; it could also reflect helix-fit-quality or
anisotropy changes). H2 holds at *every* layer. **Where it peaks moves with the model:**

| model | sharing peak | profile |
|---|---|---|
| Qwen2.5-7B | ~L14 / 28 (mid) | single mid hump |
| Mistral-Nemo-Base | ~L22 / 40 (mid-late) | broad plateau L18–28 |
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
words and Devanagari digits — **the shared subspace is sufficient to drive the answer regardless of
surface form** — while an equal-dimension random subspace does essentially nothing. This holds
against a **norm-matched** random control (`run_necessity.py` interchange: subspace_shift ≫
matched_random, e.g. Qwen es_word 1.36 vs 0.06, devanagari 1.54 vs 0.08) — so the effect is the
*specific helix directions*, not just "a large enough perturbation." Replicated on Mistral-Nemo
(weaker magnitudes, same pattern).

> Scope: this passes **isotropic** and **norm-matched** random controls. It does *not* fully settle
> the Makelov concern (a selected subspace can act through a parallel pathway) — that needs
> covariance/sensitivity-matched interchange controls (☐ pending) plus the necessity evidence below.

**Caveat — across-layer causal localization is deliberately *not* a headline.** Raw transport
magnitude isn't comparable across layers (an earlier intervention propagates through more layers →
bigger logit shift regardless of sharing). The layer-normalized `subspace/full` metric
(`run_transport_sweep.py`) fixes this for language forms but breaks for byte-fragmented scripts
(full-patch isn't a clean ceiling there). So the **single-layer** transport is the core H3 claim;
the across-layer version is at best a language-forms supplement / future work (proper causal
tracing).

### H3 (necessity): the model *relies* on the shared subspace — whole-span, and model-dependently
Sufficiency shows the circuit *can* read an injected direction; necessity asks whether the model
*relies* on the shared subspace. `run_necessity.py` mean-ablates the `en_digit`-fit helix subspace
from a source number and measures the (restricted digit-choice) accuracy drop, against three nulls —
Haar-random, **covariance-matched** (random subspace in the top activation PCA), and
**shuffled-Fourier** (a helix fit through the same pipeline on shuffled labels). Two lessons, both
from the review:

**1. Multi-token forms must be ablated over the *whole span*, not the last token.** Number-words are
1.5–2 tokens; ablating only the last fragment leaves most of the value intact. Qwen helix-ablate
accuracy vs the *strong* shuffled-Fourier null:

| Qwen form | last-token (helix vs shuf) | **whole-span** (helix vs shuf) |
|---|---|---|
| es_word | 0.54 vs 0.54 — *not* helix-specific | **0.33 vs 0.43** — helix-specific |
| fr_word | 0.62 vs 0.69 | **0.38 vs 0.50** |
| de_word | 0.79 vs 0.80 — nothing | **0.29 vs 0.50** |

At the last token, language-word "necessity" is indistinguishable from the matched nulls. Under
whole-span ablation it becomes clearly helix-specific. So the earlier "script > language necessity
gradient" was largely a **last-fragment measurement artifact** — Spanish/French/German numbers *do*
rely on the shared subspace. (Ablating the token *after* the number does ~nothing: a clean negative
control.)

**2. Reliance is model-dependent.** Under whole-span ablation Qwen shows helix-specific necessity
across scripts *and* languages. **Mistral-Nemo does not** — even whole-span, and across a full
ablation-layer sweep, its cross-form necessity vanishes on held-out data (`run_ablation_sweep.py`
discovery/test split: Mistral devanagari Δ 0.25→**0.08**, es_word 0.25→**0.00**; only en_digit holds
at 0.25). Mistral relies on the shared subspace mainly for English digits and is otherwise redundant.

The nulls do real work: covariance-matched ablation removes *more* activation energy than the helix
yet leaves arithmetic intact, so the effect is the specific helix directions, not removed energy.

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
  ablation is required for multi-token forms and *changed the finding* (language necessity is real).
  Token counts are recorded.
- ✅ **Matched controls — ADDED.** Ablation is tested against covariance-matched and shuffled-Fourier
  nulls (not just Haar-random), with removed-energy reported; interchange uses a norm-matched random
  control; per-seed curves are kept (no rounding) with mean±std.
- ◐ **"Restricted digit-choice accuracy," not arithmetic accuracy.** Renamed throughout; the readout
  is still argmax over the ten 0–9 tokens on single-digit sums. ☐ *Open:* extend to multi-digit,
  subtraction, comparison, full-continuation likelihood, word-form outputs, and same-representation
  operand+answer (main-conference reach).
- ✅ **Reproducibility.** Compact result JSONs (model revision hash, intervention norms, per-seed
  curves) are now committed (`.gitignore` keeps only large figures/caches out).
- ☐ **Statistical power.** ~24–40 cases/form; report bootstrap CIs and expand the case set before
  final figures.

## Roadmap
- [x] Step 1 — reproduce the helix fit per form
- [x] Step 2 — cross-form subspace alignment + Procrustes-CV + CKA (the go/no-go signal)
- [x] Local validation on Qwen2.5-1.5B: controls calibrated, H2 ordering visible
- [x] Tokenizer-confound check: language drop robust across `last`/`mean`/`prompt_last`; primary readout pinned to `mean`
- [x] Tooling: per-layer sweep, cross-run aggregator, tokenization diagnostic
- [x] **Real run — Qwen2.5-7B**: H2 (graded invariance) replicated, sharing localized to a mid band
- [x] **Step 3 sufficiency — causal transport** + **norm-matched** control (Qwen + Mistral-Nemo)
- [x] **Step 3 necessity — whole-span ablation vs matched nulls** (cov-matched + shuffled-Fourier): model-dependent (Qwen broad, Mistral English-digits)
- [x] **Universality — Aya-23-8B** (representational) + **Mistral-Nemo-Base** (representational + causal): H2 replicates; localization family-specific
- [x] **#6 pairwise matrix + clean H2 contrasts** (reference-form confound fixed); #7 geometry↔behavior found weak (dropped as headline)
- [x] **External-review hardening**: bug fixes; matched controls; multi-token interventions; held-out splits; committed result JSONs; **read-layer claim dropped**
- [ ] **Universality — Llama-3.1-8B (base)** (blocked on HF gated-repo approval; the original helix model) — repr. + causal
- [ ] **Extend eval** (main-conf reach): multi-digit, subtraction, comparison, word-form outputs, same-representation operand+answer; bootstrap CIs + more cases
- [ ] Optional temporal claim: causal tracing / path patching (only if pursuing the representation-vs-use question)
- [ ] Time arm — dates/years (DateAugBench has format-invariance puzzles, 2505.16088)
- [ ] Write-up — figures: pairwise heatmap, whole-span necessity (matched nulls), transport (norm-matched), layer sweep; vs FARS (2605.09496) + universal-numbers (2510.26285)

## Key references
- Kantamneni & Tegmark, *LLMs Use Trigonometry to Do Addition* — [2502.00873](https://arxiv.org/abs/2502.00873) ([code](https://github.com/subhashk01/LLM-addition))
- Engels et al., *Not All Features Are One-Dimensionally Linear* — [2405.14860](https://arxiv.org/abs/2405.14860)
- Gurnee & Tegmark, *LMs Represent Space and Time* — [2310.02207](https://arxiv.org/abs/2310.02207)
- *Separating Tongue from Thought* (cross-lingual concept patching) — [2411.08745](https://arxiv.org/abs/2411.08745)
- *Effect of Scripts and Formats on LLM Numeracy* — [2601.15251](https://arxiv.org/abs/2601.15251)
- *Language Models Learn Universal Representations of Numbers* (universal across models, disclaims cross-lingual) — [2510.26285](https://arxiv.org/abs/2510.26285)
- *FARS — Format-Agnostic Reasoning Subspaces* (closest adjacent method; general concepts, not the helix) — [2605.09496](https://arxiv.org/abs/2605.09496)
- Makelov et al., *An Interpretability Illusion for Subspace Activation Patching* — [2311.17030](https://arxiv.org/abs/2311.17030)
