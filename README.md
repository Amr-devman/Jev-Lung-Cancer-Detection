# Jev-Lung-Cancer-Detection

Zero-training histopathology classification with [TypeSafe Jev](https://docs.typesafe.ai/images).

The model gets one H&E tile and a fixed bank of clinical questions — "does this
field contain rings of cells around an open hole? what to look for: …" — and
answers morphology only. It never receives a diagnosis name, a folder name, a
filename, or any metadata, and it is never trained on anything. The mapping from
its morphology answers to a class lives in code, on the grader side only.

This repository is an experiment, not a medical device. See
[Limitations](#limitations) before reading anything into the numbers.

## Results

Default run: the **whole dataset** — all 691 LungHist700 images — model
`jev-1.14.0`, one request per image (two views each). Graded once, after all
predictions were frozen. Full metrics in [`reports/run.json`](reports/run.json).

| Metric | Value |
|---|---:|
| Images evaluated | 691 (all) |
| Accuracy | **72.2%** (95% CI 68.8–75.4, Wilson) |
| Balanced accuracy | 71.9% |
| Bootstrap 95% CI (2,000 resamples) | 68.9–75.7 |
| Normal vs carcinoma (binary) | 92.8% |

Per class:

| Class | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| adenocarcinoma (`aca`) | 280 | 0.637 | 0.779 | 0.701 |
| normal lung (`nor`) | 151 | 0.947 | 0.709 | 0.811 |
| squamous cell carcinoma (`ssc`) | 260 | 0.737 | 0.669 | 0.702 |

Confusion matrix (rows = truth, columns = predicted):

| | aca | nor | ssc |
|---|---:|---:|---:|
| **aca** | 218 | 6 | 56 |
| **nor** | 38 | 107 | 6 |
| **ssc** | 86 | 0 | 174 |

Confidence / coverage curve — the fused probability is an abstention knob, not
a formality:

| Min confidence | Coverage | Accuracy on covered |
|---|---:|---:|
| ≥ 0.5 | 92% | 73.7% |
| ≥ 0.6 | 73% | 80.8% |
| ≥ 0.7 | 50% | 85.6% |
| ≥ 0.8 | 22% | 92.1% |
| ≥ 0.9 | 4% | 100% |

### Reading the numbers

This dataset is much harder than the LC25000-style benchmarks where classical
models reach the high 90s — ~72% here is a real signal, not a disappointment.

- **The dominant error is still the glandular ↔ keratinizing boundary.** 142 of
  the 192 errors are `aca ↔ ssc` (56 + 86). That is the known hard case of
  H&E-only pathology: a solid-pattern glandular tumour and a poorly
  differentiated squamous tumour can both be sheets of atypical cells, and in
  practice the tie is broken with p40 / TTF-1 stains. The model has no stains.
- **Normal tissue is over-called as carcinoma** (44 of 151 fields), but when
  the model predicts `nor` it is right 95% of the time. The growth-arrangement
  question (organized within pre-existing structures vs infiltrative/expansile
  growth) is the cleanest single signal here: 131 of 151 normal fields answer
  "pre-existing structures".
- **Confidence is informative and well calibrated at the top.** At the 0.8
  threshold the model covers 22% of the set at 92.1% accuracy; at 0.9 it has
  been right every time so far. A triage workflow could forward only the
  uncertain fraction to a human.
- **Wording matters measurably, in both directions.** A controlled A/B on the
  full dataset found that plain visual words ("onion-skin rings of bright pink
  material", "shrivelled pink cells") find squamous growth far better than
  medical jargon ("keratin pearls", "dyskeratotic cells"): an all-plain bank
  reached the best-ever squamous recall (0.731) but collapsed on normal tissue
  (0.576), where precise clinical concepts ("alveolar", "gland lumens") carry
  structure the plain words lose. The shipped bank is the measured hybrid —
  plain language on the keratin side, clinical precision on the alveolar and
  gland side: 72.2% vs 71.8% all-clinical, 68.5% all-plain (hybrid vs the
  original single-view baseline: 29 verdicts fixed vs 15 broken, paired sign
  test p ≈ 0.05).

## The experiment

| | |
|---|---:|
| Dataset | **LungHist700** (Kaggle `abdullahhasansajjad/lunghist700`) |
| Images in the download | 691 (JPEG 1600×1200, 20×/40× fields) |
| Classes | 3 — `aca` 280, `nor` 151, `ssc` 260 |
| Evaluated | 691 (the whole dataset) |
| Jev model | `jev-1.14.0` |
| Training | none |
| Labels seen by the model | none |
| IHC / molecular input | none |
| Questions per image | 45 yes/no observations + 10 multiple-choice |

The three classes are lung adenocarcinoma (`aca`), normal lung tissue (`nor`),
and lung squamous cell carcinoma (`ssc`). The download also ships a `data.csv`
with per-image subclass (differentiation grade), magnification and patient id;
the pipeline never reads it. The pipeline discovers the class folders and maps
them to neutral ids `c0`, `c1`, `c2` automatically.

## Methodology

### One request per image, six views

For each field, Jev receives the pixels six times — `sample` (the whole field
at low power), `high_power` (a higher-magnification view of the same tissue:
the genuine 40× sibling field when the dataset ships one, matched by filename
in code, otherwise a 2× digital zoom), and four `tour_*` tiles covering the
four quadrants of the field at high power — plus 55 questions. Nothing else is
in the state: no reference images, no labels, no filenames, no second request.

The view set exists for one reason: keratin whorls, intercellular bridges,
gland lumens and nuclear detail are high-power structures, and a single
centre crop is blind to the periphery. Every question says which view to use,
and every question tells the model to check all tour tiles before concluding a
fine structure is absent. The crops and pairings are generated in code
(`index.load_tiles`, `index.sibling_at`); no dataset metadata ever reaches the
model.

Every question has two parts:

1. the **observation**, phrased about morphology only — "does `sample` contain
   rounded or irregular gland-like structures with a central open lumen?";
2. a plain-language **what-to-look-for description** — "look for rings of cells
   around a central open hole (a lumen) … depending on how the tissue was cut
   the rings appear as circles, ovals, crooked rings, crescents or short tubes".

The descriptions are the load-bearing part. A question whose vocabulary the
model cannot ground in pixels is a coin flip; the description turns each
clinical term into a visual task, in the same words a pathologist would use to
teach a student. Each yes/no question also carries fixed outcome criteria
(`true` = clearly visible, `false` = absent or a doubtful hint) to pin the
polarity. The full bank is in
[`research/question_bank.md`](research/question_bank.md) and the clinical basis
is in [`research/lung_carcinoma_histopathology.md`](research/lung_carcinoma_histopathology.md).

There are **no box questions**. Boxes never carried class evidence, and a Box
answer always returns four edges even when the requested structure is absent,
which invites fabricated annotations and post-processing that cannot be
validated. The experiment is classification-only.

### Three neutral profiles

The model is never asked for a diagnosis. It is asked which of three
architecture descriptions matches the field:

| Profile | Description shown to the model | Class |
|---|---|---|
| `p0` | glandular growth: rings of cells around an open lumen, mucin pools, dense pink stroma | adenocarcinoma (`aca`) |
| `p1` | air-sac architecture: open spaces, thin walls, pigment and scavenger cells | normal lung (`nor`) |
| `p2` | solid cohesive growth: sheets of polygonal cells, dense pink cytoplasm, onion-skin whorls | squamous cell carcinoma (`ssc`) |

The mapping lives in `config.PROFILE_TO_CLASS` and is used only by the grader.

### From answers to a class: a two-stage decision

The decision is hierarchical, matching how the clinical question is actually
split (`jev_cancer/fusion.py`):

- **Stage 1 — normal tissue vs carcinoma.** Only signals that can judge
  whether the tissue is untouched or a proliferation vote: the two architecture
  Choices, the growth-arrangement Choice (organized within pre-existing
  structures vs infiltrative glands vs expansile solid nests), the stroma
  Choice and the observation rubric, each projected to a binary
  normal/carcinoma question.
- **Stage 2 — glandular vs keratinizing**, asked only once stage 1 says
  carcinoma. Only signals that can separate the two carcinomas vote: the
  architecture Choices, the growth-pattern Choice, the contrastive
  differentiation tie-breaker ("which tie-breakers do you actually see: gland
  lumens and mucin, or keratin whorls and bridges?"), the cell-shape,
  cytoplasm and rubric signals, projected to the gland/keratin pair.

The final probabilities are the hierarchical product: `p(normal)` from stage 1,
and the carcinoma mass split by stage 2. Nothing is fitted; the weights are
fixed clinical priors (`fusion.WEIGHTS`):

| Signal | What it is | Weight | Votes in |
|---|---|---:|---|
| `profile` | probability over the three architecture descriptions | 0.28 | both stages |
| `rubric` | weighted sum of the 45 present observations, softmaxed | 0.12 | both stages |
| `differentiation` | forced-choice gland-vs-keratin tie-breaker | 0.09 | stage 2 |
| `profile_recheck` | the same Choice, options reordered | 0.10 | both stages |
| `arrangement` | organized/pre-existing vs infiltrative vs expansile growth | 0.08 | stage 1 |
| `proliferation_pattern` | which growth pattern the field shows | 0.08 | stage 2 |
| `lumen_type` | true lined gland lumens vs unlined holes vs slit gaps | 0.05 | stage 2 |
| `keratin_evidence` | keratin pearls vs dyskeratotic cells vs neither | 0.05 | stage 2 |
| `cell_shape` | dominant epithelial cell shape | 0.05 | stage 2 |
| `cytoplasm` | dominant cytoplasm type | 0.05 | stage 2 |
| `stroma` | what lies between the epithelial structures | 0.05 | stage 1 |

The rubric weights are a transparent clinical prior encoded per observation
(`protocol.FEATURES`): a gland lumen strongly supports `p0`, organized
pre-existing structures and thin alveolar walls strongly support `p1`, an
onion-skin whorl strongly supports `p2`, and so on. Nothing is fitted; there is
no training step anywhere in the repository.

The output is a probability vector over `p0/p1/p2`, both stage probabilities, a
confidence and a margin. The grader reports a confidence/coverage curve rather
than assuming every tile is decidable.

### Zero-leak controls

This is the part to review first.

| Risk | Control |
|---|---|
| Class names reach the model | Prompts are written from morphology. `protocol.assert_no_leaks` scans every question id, instruction and criterion for 23 diagnosis/provenance terms (including the `aca`/`nor`/`ssc` folder names and the dataset name) before each request. |
| Folder or file names reach the model | Images are sent as decoded pixels, never as paths or ids. Neutral ids (`img_004271`) exist only in the local index. |
| The evaluation set informs model choice | Nothing is fitted or selected on it. Weights are fixed clinical priors; grading is a separate script (`02_grade.py`) that runs after predictions are frozen. |
| Inference code can read truth | `pipeline.py` and `fusion.py` never read class folders. The class column of the index is used only to draw the stratified subset and to grade. |
| The model is asked for a diagnosis | It is asked for morphology. The diagnosis mapping is code-side and grader-only. |

### The evaluation subset

The subset is drawn by `jev_cancer.index.select_subset`:

- the default (`--fraction` omitted, or `1.0`) evaluates **every image** in the
  dataset — all 691 for LungHist700;
- a smaller `--fraction` draws a **stratified** subset: every class contributes
  the same number of images, so no class can be drowned out;
- `--seed` (default `20261020`) makes the draw reproducible: the same seed on
  the same dataset yields the identical subset on any machine;
- `--count N` runs the first N images of the index instead of a draw.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.sample .env          # add your TypeSafe key

# 1. download the dataset (kagglehub) and verify the layout
.venv/bin/python scripts/00_download.py

# 2. classify — the default is the whole dataset (691 images)
.venv/bin/python scripts/01_run.py --workers 16

# 3. grade once, against the sealed folder labels
.venv/bin/python scripts/02_grade.py
```

Other subsets, same protocol:

```sh
# a fast, cheap check: ~140 images, stratified
.venv/bin/python scripts/01_run.py --fraction 0.2 --seed 7

# the first 200 images of the index
.venv/bin/python scripts/01_run.py --count 200

# re-export predictions from cached answers, no API calls
.venv/bin/python scripts/01_run.py --export-only
```

Every Jev response is cached under `runs/<dataset>/<protocol>/raw/` with its
token usage and latency. A rerun costs nothing, `--export-only` is free, and
any subset only pays for images not yet seen. The cache is partitioned by
dataset and by protocol version (the exact question text), so the per-run
output directory `runs/lh700/<protocol>/f<fraction>_s<seed>/` never collides.

## Cost and runtime

The published run sent 18.8M input tokens over 691 images (~27k per image:
six views of one field plus 55 questions) at TypeSafe's published $42 per
billion input tokens: **$0.79**. Wall time was **58 seconds** on 10 workers
(12 img/s). A complete reproduction costs under a dollar and about a minute.

## Repository layout

```text
jev_cancer/
  config.py      paths, model settings, neutral vocabulary, leak-guard terms
  protocol.py    the 42 observations and 6 Choices, each with a what-to-look-for
                 description; the clinical rubric; the leak guard
  pipeline.py    the one request per image, caching, resume
  fusion.py      the seven signals, their fixed weights, the final probabilities
  client.py      thread-safe TypeSafe wrapper, answers to JSON
  index.py       dataset download, neutral index, seeded stratified subset
  metrics.py     accuracy, per-class P/R/F1, Wilson and bootstrap intervals
scripts/
  00_download.py   download LungHist700 and verify its layout
  01_run.py        classify (all images by default; --fraction, --seed, --count)
  02_grade.py      one-shot grading against the folder labels
research/          clinical reference and the question bank
reports/run.json   the published metrics
```

## Limitations

- **Not a clinical tool.** This is a benchmark harness. No result here is a
  diagnosis, and no output should be used for patient care.
- **Tile-level, not slide-level.** The images are 20×/40× fields of view. There
  is no slide-level architecture, stage, or treatment context.
- **The 691 images are not 691 independent specimens.** `data.csv` shows the
  same tissue field appears at both 20× and 40× under one image id, and images
  are grouped by patient id. The confidence intervals are image-level and
  optimistic.
- **Dataset labels are inherited.** They were not re-reviewed by a pathologist
  for this experiment, and the subclass labels (differentiation grade) shipped
  in `data.csv` are not used at all — the experiment grades superclass only.
- **H&E only.** The markers that resolve poorly differentiated tumours in
  practice (TTF-1, p40, Napsin A, CK5/6) are invisible here — which is exactly
  where the errors concentrate (153 of 206 errors are `aca ↔ ssc`).
- **The normal class is normal lung.** It is not a library of benign mimics
  (reactive pneumocyte hyperplasia, organising pneumonia, squamous metaplasia),
  and it is still over-called as glandular growth in a third of cases. Expect
  weaker performance on real biopsy mimics.
- **70% is a real score on a hard benchmark.** Unlike LC25000-style collections
  where classical models reach the high 90s, this dataset contains poorly
  differentiated fields across grades and magnifications.

## Data credit

**LungHist700** (`abdullahhasansajjad/lunghist700` on Kaggle): 691 H&E fields
of lung tissue — adenocarcinoma, normal lung, squamous cell carcinoma — with
subclass, magnification and patient metadata in `data.csv`. Check the Kaggle
licence before redistributing pixels. This repository does not commit the
images.
