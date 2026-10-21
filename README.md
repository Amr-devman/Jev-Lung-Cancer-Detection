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
`jev-1.14.0`, one request per image. Graded once, after all predictions were
frozen. Full metrics in [`reports/run.json`](reports/run.json).

| Metric | Value |
|---|---:|
| Images evaluated | 691 (all) |
| Accuracy | **70.2%** (95% CI 66.7–73.5, Wilson) |
| Balanced accuracy | 69.6% |
| Bootstrap 95% CI (2,000 resamples) | 66.7–73.5 |
| Normal vs carcinoma (binary) | 92.3% |

Per class:

| Class | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| adenocarcinoma (`aca`) | 280 | 0.615 | 0.771 | 0.685 |
| normal lung (`nor`) | 151 | 0.971 | 0.669 | 0.792 |
| squamous cell carcinoma (`ssc`) | 260 | 0.712 | 0.646 | 0.677 |

Confusion matrix (rows = truth, columns = predicted):

| | aca | nor | ssc |
|---|---:|---:|---:|
| **aca** | 216 | 3 | 61 |
| **nor** | 43 | 101 | 7 |
| **ssc** | 92 | 0 | 168 |

Confidence / coverage curve — the fused probability is an abstention knob, not
a formality:

| Min confidence | Coverage | Accuracy on covered |
|---|---:|---:|
| ≥ 0.5 | 94% | 71.5% |
| ≥ 0.6 | 76% | 76.1% |
| ≥ 0.7 | 57% | 80.8% |
| ≥ 0.8 | 33% | 83.8% |
| ≥ 0.9 | 6% | 95.5% |

### Reading the numbers

This dataset is much harder than the LC25000-style benchmarks where classical
models reach the high 90s — 70% here is a real signal, not a disappointment.

- **The dominant error is still the glandular ↔ keratinizing boundary.** 153 of
  the 206 errors are `aca ↔ ssc` (61 + 92). That is the known hard case of
  H&E-only pathology: a solid-pattern glandular tumour and a poorly
  differentiated squamous tumour can both be sheets of atypical cells, and in
  practice the tie is broken with p40 / TTF-1 stains. The model has no stains.
- **Normal lung is over-called as glandular growth** (43 of 151 normal tiles →
  `aca`, recall 0.669) while being almost never confused with squamous (7).
  Cellular normal fields — bronchial walls, macrophage-filled spaces, reactive
  epithelium — read as crowded glandular growth. The reverse failure is rare:
  when the model predicts `nor` it is right 97% of the time.
- **Confidence is informative.** At the 0.7 threshold the model covers 57% of
  the set at 80.8% accuracy; at 0.9 it is 95.5% accurate on the 6% it is
  certain about. A triage workflow could forward only the uncertain fraction to
  a human.

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
| Questions per image | 42 yes/no observations + 6 multiple-choice |

The three classes are lung adenocarcinoma (`aca`), normal lung tissue (`nor`),
and lung squamous cell carcinoma (`ssc`). The download also ships a `data.csv`
with per-image subclass (differentiation grade), magnification and patient id;
the pipeline never reads it. The pipeline discovers the class folders and maps
them to neutral ids `c0`, `c1`, `c2` automatically.

## Methodology

### One request per image

For each tile, Jev receives the pixels as `sample` and 48 questions. Nothing
else is in the state: no reference images, no labels, no filenames, no second
request. Every question has two parts:

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

### From answers to a class

Seven signals are computed from the one response and combined by a weighted
mean with fixed clinical-prior weights (`jev_cancer/fusion.py`):

| Signal | What it is | Weight |
|---|---|---:|
| `profile` | probability over the three architecture descriptions | 0.35 |
| `profile_recheck` | the same Choice, options reordered, "architecture not staining" framing | 0.15 |
| `proliferation_pattern` | which growth pattern the field shows, soft-mapped to profiles | 0.15 |
| `rubric` | weighted sum of the 42 present observations, softmaxed | 0.15 |
| `cell_shape` | dominant epithelial cell shape, soft-mapped | 0.08 |
| `cytoplasm` | dominant cytoplasm type, soft-mapped | 0.07 |
| `stroma` | what lies between the epithelial structures, soft-mapped | 0.05 |

The rubric weights are a transparent clinical prior encoded per observation
(`protocol.FEATURES`): a gland lumen strongly supports `p0`, thin alveolar
walls strongly support `p1`, an onion-skin whorl strongly supports `p2`, and so
on. Nothing is fitted; there is no training step anywhere in the repository.

The output is a probability vector over `p0/p1/p2`, a confidence and a margin.
The grader reports a confidence/coverage curve rather than assuming every tile
is decidable.

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

The published run sent 7.3M input tokens over 691 images (~10.6k per image:
one 1600×1200 image plus 48 questions) at TypeSafe's published $42 per billion
input tokens: **$0.31**. Wall time was **42 seconds** on 16 workers (16 img/s).
A complete reproduction costs about a third of a dollar and half a minute.

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
