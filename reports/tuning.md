# Question-tuning experiment

Fine-tuning the clinical question bank by iteration: each iteration tests one
variant of the questions on a fixed random panel of 10 images, logs binary and
per-class accuracy, and the next variant is chosen from the observed errors.
Maximum budget: 100 iterations / 1,000 image-tests. **Spent: 48 iterations,
836 image-tests.** Raw answers are cached per variant in `runs/tuning/`; the
machine-readable trend is `reports/tuning/log.jsonl`; the harness is
`experiments/tune.py` and the variants are `experiments/variants.py`.

## The panel

10 random images, stratified 4/3/3 across the three classes, drawn once with
seed `20261021` (`experiments/panel.json`). It is fixed so that variants are
compared on identical pixels — only the questions change. Two things guard
against overfitting to it:

- a **holdout panel** (seed `20261022`, 10 images) that variants are only
  scored on, never designed against;
- a **validation panel** (seed `20261023`, 99 images, 33 per class) for the
  final comparison.

With n=10, one image is 10 percentage points, so panel-A/B scores are read as
paired comparisons ("which variant gets which image right"), not as estimates.

## Literature behind the variants

| Source | Principle applied |
|---|---|
| Prakash et al., *Enhancing Clinical Reasoning in Medical VLM through Structured Prompts* (medRxiv 2025) | Clinically informed, structured prompts beat generic ones; templates that guide visual feature extraction before judgment improve accuracy and generalization. → decision-rule phrasing (v16–v19), tie-breaker-first framing (v08+). |
| VLM answer-bias analyses ("yes/no questions default to agreement"; presence-question answer bias) | Independent yes/no presence questions over-fire "yes". → quantified criteria requiring ≥2 clear examples (v10, v14, v18, v21). |
| Forced-choice vs single-option questioning ("a question that names both options and forces a choice uses it well") | Contrastive forced choice discriminates better than isolated presence questions. → the differentiation tie-breaker Choice (v09, v13, v22, v23). |
| MI-Zero / CONCH / CPLIP prompt work in computational pathology | Descriptive morphology prompts outperform label-like prompts; descriptions must enumerate the visual attributes. → the what-to-look-for descriptions (already in the base bank) and attribute-specific profile texts. |
| Question-decomposition literature (IdealGPT, medical VQA decomposition) | Perception questions before the final judgment; the final question can condition on earlier observations. → Noul-first ordering (already), self-conditioned profile (v24, v25). |

Every variant passes `protocol.assert_no_leaks` before any request: the tuning
loop cannot smuggle diagnosis words into prompts even to win points.

## Findings: which questions carry the accuracy

1. **Binary accuracy is solved.** Every variant scores 1.000 benign-vs-carcinoma
   on every panel. All the signal is in the adeno-vs-squamous split, and every
   tunable question matters only there.
2. **The atomic observations carry the class signal; the holistic Choices are
   the weak link.** The ablations (iterations 2–8) show:
   - dropping any one observation group (adequacy, background, cytology,
     architecture) changes nothing on the tuning panel;
   - the **rubric alone (42 Nouls, no Choices) scores 0.90 vs the full bank's
     0.80**;
   - the **Choices alone score 0.80** — they make the same errors as the full
     bank, and they are what pull hard adeno cases to squamous. On the
     error tiles the Choice voted "solid sheets dominate" while the Noul
     `a_glands` had already seen the gland rings (0.81).
3. **Descriptions should define differentiation, not dominant texture.** The
   single biggest jump (iteration 9: 0.80 → 1.00 on the tuning panel) came from
   rewriting the three profile descriptions: the glandular description claims
   solid fields with gland evidence, and the keratinizing description *requires*
   a keratinizing tie-breaker (whorls, bridges, dense keratinizing cytoplasm) —
   solid sheets alone no longer qualify.
4. **One-sided fixes overfit; symmetric rules generalize.** v08's "glands
   anywhere" clause fixed both tuning-panel adenos but flipped a holdout
   squamous tile on mucin alone (iteration 14). The symmetric rule — p0 needs
   clear gland rings in ≥2 places (mucin alone is not enough), p2 needs a
   keratinizing tie-breaker — plus strict ≥2-example criteria on the two
   hallucination-prone keratin observations (v21) keeps both panels intact.
5. **The model contradicts itself at the margin — question wording alone cannot
   fully fix it.** On one hard tile the forced-choice tie-breaker answers
   "gland differentiation (0.59)" while the architecture Choice answers
   "keratinizing (0.60)". Carrying the tie-breaker as extra evidence (v22, v23)
   moves the margin from 0.50/0.47 to 0.50/0.50 and flips the holdout's
   twin tile — the two tiles are perceptually near-identical to the model
   (differing mainly in `a_glands` 0.80 vs 0.41). This is the documented
   H&E limit: solid-pattern glandular vs poorly differentiated keratinizing
   growth is resolved by immunohistochemistry in practice, not by wording.
6. **Self-conditioning and decision-rule instructions did not help** (v16–v19,
   v24–v25): the model's Choice answers barely respond to instruction framing
   and mostly respond to the *option descriptions* themselves. Wording effort
   belongs in the criteria text, not the instruction prefix.

## Score trend

Panels: `A-10` = tuning panel (seed 20261021), `B-10` = holdout (20261022),
`holdout-99` = validation (20261023, 33 per class). One row per iteration; the
"what changed" column is the variant note.

| iter | variant | panel | overall | binary | adeno | benign | squamous | what changed |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | v00_base | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | baseline: current protocol (42 Nouls + 6 Choices, described) |
| 2 | v01_no_adequacy | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ablation: drop the 3 adequacy gates (q_*) |
| 3 | v02_no_background | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ablation: drop the 5 background observations (b_*) |
| 4 | v03_no_cytology | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ablation: drop the 16 cytology observations (c_*) |
| 5 | v04_no_architecture | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ablation: drop the 18 architecture observations (a_*) |
| 6 | v05_rubric_only | A-10 | 0.90 | 1.000 | 0.75 | 1.00 | 1.00 | ablation: no Choices at all — the 42-observation rubric alone |
| 7 | v06_choices_only | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ablation: no Nouls at all — the 6 Choices alone |
| 8 | v07_no_recheck | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ablation: drop the reordered profile_recheck |
| 9 | v08_sharp_profiles | A-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | profiles judge differentiation not area; p2 requires a keratinizing tie-breaker |
| 10 | v09_tiebreak | A-10 | 0.90 | 1.000 | 0.75 | 1.00 | 1.00 | add a forced-choice differentiation tie-breaker (gland vs keratin) |
| 11 | v10_quantified | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | ≥2 clear examples required on the tie-breaker observations |
| 12 | v11_combined | A-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | sharpened profiles + tie-breaker + quantified criteria |
| 13 | v00_base | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | baseline on the holdout panel |
| 14 | v08_sharp_profiles | B-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | v08 overfits: flips a squamous tile on mucin alone |
| 15 | v12_balanced | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | symmetric rule: p0 needs clear glands AND no keratin; p2 the converse |
| 16 | v12_balanced | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | symmetric rule holds the holdout |
| 17 | v13_balanced_tiebreak | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v12 + tie-breaker choice at 0.08 |
| 18 | v13_balanced_tiebreak | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | v12 + tie-breaker choice at 0.08 |
| 19 | v14_balanced_strict | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v12 + strict ≥2 examples on a_whorl/a_spines |
| 20 | v14_balanced_strict | B-10 | 0.90 | 1.000 | 0.75 | 1.00 | 1.00 | strict criteria alone hold neither panel |
| 21 | v15_balanced_both | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v12 + tie-breaker + strict |
| 22 | v15_balanced_both | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | v12 + tie-breaker + strict |
| 23 | v16_rule_profile | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | profile questions become ordered decision rules |
| 24 | v16_rule_profile | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | ordered decision rules hold the holdout |
| 25 | v17_rule_tiebreak | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v16 + tie-breaker at 0.08 |
| 26 | v17_rule_tiebreak | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | v16 + tie-breaker at 0.08 |
| 27 | v18_rule_strict | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v16 + strict keratin criteria |
| 28 | v18_rule_strict | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | v16 + strict keratin criteria |
| 29 | v19_rule_all | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v16 + tie-breaker + strict |
| 30 | v19_rule_all | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | v16 + tie-breaker + strict |
| 31 | v20_precise_glands | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v08 split: p0 needs gland rings in ≥2 places, not mucin alone |
| 32 | v20_precise_glands | B-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | the p2 clause alone flips the holdout squamous tile |
| 33 | v21_precise_strict | A-10 | 0.90 | 1.000 | 0.75 | 1.00 | 1.00 | v20 + strict ≥2-example criteria on a_whorl/a_spines |
| 34 | v21_precise_strict | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | v20 + strict keratin criteria holds both panels |
| 35 | v22_tiebreak_015 | A-10 | 0.90 | 1.000 | 0.75 | 1.00 | 1.00 | v21 + tie-breaker carried at 0.15 |
| 36 | v22_tiebreak_015 | B-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | tie-breaker weight flips the twin tile |
| 37 | v23_tiebreak_025 | A-10 | 0.90 | 1.000 | 0.75 | 1.00 | 1.00 | v21 + tie-breaker at 0.25 |
| 38 | v23_tiebreak_025 | B-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | same trade at higher weight |
| 39 | v24_self_conditioned | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | profile decides from the model's own observation answers |
| 40 | v24_self_conditioned | B-10 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | self-conditioning holds the holdout |
| 41 | v25_self_tiebreak | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v24 + tie-breaker at 0.15 |
| 42 | v25_self_tiebreak | B-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | v24 + tie-breaker at 0.15 |
| 43 | v00_base | holdout-99 | 0.89 | 1.000 | 0.82 | 1.00 | 0.85 | baseline on 99 validation images |
| 44 | v08_sharp_profiles | holdout-99 | 0.89 | 1.000 | 0.88 | 1.00 | 0.79 | gains adeno, loses squamous — the one-sided fix confirmed |
| 45 | v21_precise_strict | holdout-99 | **0.91** | 1.000 | 0.85 | 1.00 | 0.88 | **best: +2pp overall, +3pp adeno, +3pp squamous** |
| 46 | v26_mucin_concession | A-10 | 0.80 | 1.000 | 0.50 | 1.00 | 1.00 | v21 + mucin qualifies alongside gland evidence |
| 47 | v26_mucin_concession | B-10 | 0.90 | 1.000 | 1.00 | 1.00 | 0.67 | the concession reopens the squamous leak |
| 48 | v26_mucin_concession | holdout-99 | 0.90 | 1.000 | 0.88 | 1.00 | 0.82 | rejected: 5 squamous tiles flip |

## The winner (and its rejection at scale)

`v21_precise_strict` won the panel experiment:

1. **Profile descriptions judge differentiation, not dominant texture.** `p0`
   claims a solid field when well-formed gland rings with lumens are clearly
   present in at least two separate places (mucin pools alone do not qualify);
   `p2` requires at least one keratinizing tie-breaker — solid sheets alone do
   not qualify.
2. **Strict ≥2-example outcome criteria** on `a_whorl` and `a_spines`, the two
   hallucination-prone tie-breaker observations.

It scored 0.91 vs the baseline bank's 0.89 on the 99-image validation panel —
a 2-image difference, explicitly flagged below as "a direction, not an effect
size".

### Post-hoc scale test (the decisive one)

Both banks were then run over the **same first 2,000 images of the index**
(`scripts/01_run.py --count 2000`), graded with the same grader:

| Bank | Overall | Adeno | Benign | Squamous |
|---|---:|---:|---:|---:|
| pre-tuning (`804c139028d5a355`) | **0.9125** | 0.830 | 0.998 | 0.914 |
| v21 tuned (`d5ff8dc8c896bf8b`) | 0.8965 | 0.852 | 0.998 | 0.845 |
| delta | **−1.6pp** | +2.2pp | 0 | **−6.9pp** |

83 profiles changed: 25 old errors fixed, 57 newly broken (exact paired sign
test p ≈ 0.0005). The regression is real, not noise.

**Verdict: v21 was rejected and the shipped bank is the pre-tuning one.** The
mechanism: v21's "p2 requires a keratinizing tie-breaker" plus the strict
≥2-example criteria suppress keratin evidence, trading +2.2pp adeno for −6.9pp
squamous. The panel loop selected for exactly the two tiles it was designed
against and paid for it everywhere else. Lesson recorded in the README: wording
changes at this boundary move both classes at once and must be selected on
thousands of images, not panels.

Metrics files: `reports/run_head2000_pretuning.json` (0.9125) and
`reports/run_head2000_v21.json` (0.8965).

## Caveats

- The two 10-image panels are for paired comparison only; the +2pp claim rests
  on the 99-image validation, where the difference is 2 images. Treat it as a
  direction, not a precise effect size.
- The tuning loop optimizes on labels from the same dataset the pipeline is
  benchmarked on. It is question wording, not training — but it is not a
  held-out-dataset result either.
- The remaining errors are concentrated in exactly the clinically hard case
  (solid glandular vs poorly differentiated keratinizing growth). Better
  wording has a ceiling there; the honest fix is immunohistochemistry, which
  this benchmark cannot supply.

## Reproduce

```sh
.venv/bin/python experiments/tune.py --variant v21_precise_strict            # panel A
.venv/bin/python experiments/tune.py --variant v21_precise_strict --panel-seed 20261022
.venv/bin/python experiments/tune.py --variant v21_precise_strict --panel-seed 20261023 --per-class 33
.venv/bin/python experiments/tune.py --variant all                           # re-score everything (cached)
```
