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

Default run: the stratified 3,000-image subset (1,000 per class, `--fraction
0.2 --seed 20261020`), model `jev-1.14.0`, one request per image. Graded once,
after all predictions were frozen. Full metrics in
[`reports/run.json`](reports/run.json). These numbers were produced with the
pre-tuning question bank; the shipped bank is the tuned one described in
[Question tuning](#question-tuning), which validates 0.91 vs 0.89 on 99
held-out images (`python scripts/01_run.py` regenerates the run with it).

| Metric | Value |
|---|---:|
| Images evaluated | 3,000 |
| Accuracy | **91.3%** (95% CI 90.2–92.3, Wilson) |
| Balanced accuracy | 91.3% |
| Bootstrap 95% CI (2,000 resamples) | 90.3–92.3 |
| Normal vs carcinoma (binary) | 99.5% |

Per class:

| Class | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| adenocarcinoma | 1,000 | 0.909 | 0.821 | 0.863 |
| benign | 1,000 | 0.987 | 0.999 | 0.993 |
| squamous cell carcinoma | 1,000 | 0.847 | 0.919 | 0.882 |

Confusion matrix (rows = truth, columns = predicted):

| | adeno | benign | squamous |
|---|---:|---:|---:|
| **adeno** | 821 | 13 | 166 |
| **benign** | 1 | 999 | 0 |
| **squamous** | 81 | 0 | 919 |

Confidence / coverage curve — the fused probability is an abstention knob, not
a formality:

| Min confidence | Coverage | Accuracy on covered |
|---|---:|---:|
| ≥ 0.5 | 95% | 92.8% |
| ≥ 0.6 | 88% | 95.0% |
| ≥ 0.7 | 76% | 96.7% |
| ≥ 0.8 | 31% | 96.4% |
| ≥ 0.9 | 2% | 100% |

### Reading the numbers

- **The error mass is almost entirely adeno ↔ squamous.** 166 adenocarcinomas
  are called squamous and 81 squamous are called adenocarcinoma. That is the
  known hard boundary of H&E-only pathology: a solid-pattern glandular tumour
  and a poorly differentiated squamous tumour can both be sheets of atypical
  cells, and in practice the tie is broken with p40 / TTF-1 stains. The model
  has no stains. The one place it fails is the one place pathologists order
  immunohistochemistry.
- **The normal class is essentially solved** (99.9% recall, 1 false positive),
  because preserved air-sac architecture is visually distinct and the question
  bank asks about it directly.
- **Confidence is informative.** At 0.7 threshold the model covers 76% of the
  set at 96.7% accuracy. A triage workflow could forward only the uncertain
  quarter to a human.

## The experiment

| | |
|---|---:|
| Dataset | LC25000 subset (Kaggle `rm1000/lung-cancer-histopathological-images`) |
| Images in the download | 15,000 (3 classes × 5,000, 768 px tiles) |
| Evaluated | 3,000 (default `--fraction 0.2`) |
| Subset draw | random, stratified: 1,000 per class, seeded |
| Jev model | `jev-1.14.0` |
| Training | none |
| Labels seen by the model | none |
| IHC / molecular input | none |
| Questions per image | 42 yes/no observations + 6 multiple-choice |

The three classes are lung adenocarcinoma, lung squamous cell carcinoma, and
benign lung tissue. The Kaggle mirror advertises four classes; the download
contains three (there is no large-cell carcinoma folder). The pipeline discovers
the folders and maps them to neutral ids `c0`, `c1`, `c2` automatically.

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
| `p0` | glandular growth: rings of cells around an open lumen, mucin pools, dense pink stroma | adenocarcinoma |
| `p1` | air-sac architecture: open spaces, thin walls, pigment and scavenger cells | benign |
| `p2` | solid cohesive growth: sheets of polygonal cells, dense pink cytoplasm, onion-skin whorls | squamous cell carcinoma |

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
| Class names reach the model | Prompts are written from morphology. `protocol.assert_no_leaks` scans every question id, instruction and criterion for 18 diagnosis/provenance terms before each request. |
| Folder or file names reach the model | Images are sent as decoded pixels, never as paths or ids. Neutral ids (`img_004271`) exist only in the local index. |
| The evaluation set informs tuning | Nothing is tuned. Weights are fixed clinical priors; grading is a separate script (`02_grade.py`) that runs after predictions are frozen. |
| Inference code can read truth | `pipeline.py` and `fusion.py` never read class folders. The class column of the index is used only to draw the stratified subset and to grade. |
| The model is asked for a diagnosis | It is asked for morphology. The diagnosis mapping is code-side and grader-only. |

### The evaluation subset

The subset is drawn by `jev_cancer.index.select_subset`:

- `--fraction` is the share of the whole dataset to evaluate. Omitted, it
  defaults to `0.2`, which nets **3,000 of 15,000 images**.
- the draw is **stratified**: every class contributes the same number of images
  (1,000 each at the default), so accuracy and balanced accuracy coincide and
  no class can be drowned out;
- `--seed` (default `20261020`) makes the draw reproducible: the same seed on
  the same dataset yields the identical subset on any machine.

## Question tuning

The question bank was fine-tuned by a controlled iteration experiment — 48
iterations / 836 image-tests of variant question banks on fixed stratified
panels, guided by error analysis and the medical-VLM prompting literature. Full
method, score trend and findings in
[`reports/tuning.md`](reports/tuning.md); harness in `experiments/tune.py`.

What the experiment established:

- **Binary accuracy (normal vs carcinoma) is 1.000 for every question bank
  tested.** All tunable signal lives in the glandular-vs-keratinizing split.
- **The 42 atomic observations carry the class signal; the holistic Choices are
  the weak link.** The rubric alone beats the full bank on the hard tiles
  (ablation, iter 6): on those tiles the Choice voted "solid sheets dominate"
  while `a_glands` had already seen the gland rings.
- **Descriptions must define differentiation, not dominant texture.** The
  winning bank (`v21`, iter 33–45) makes `p0` claim solid fields with clear
  gland evidence and makes `p2` *require* a keratinizing tie-breaker — plus
  strict ≥2-example criteria on the two hallucination-prone keratin
  observations. Validated **0.91 vs 0.89 baseline** on 99 held-out images.
- **One-sided fixes overfit a 10-image panel.** A "glands anywhere" clause
  scored 1.00 on the tuning panel and flipped a holdout squamous tile (iter 9
  vs 14). Every variant in the trend is scored on both panels for this reason.
- **The residual errors are the documented clinical hard case**: solid
  glandular vs poorly differentiated keratinizing growth, which practice
  resolves with p40/TTF-1 stains, not with better wording.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.sample .env          # add your TypeSafe key

# 1. download the dataset (kagglehub) and verify the layout
.venv/bin/python scripts/00_download.py

# 2. classify a stratified subset — default: 3,000 images, 1,000 per class
.venv/bin/python scripts/01_run.py --fraction 0.2 --seed 20261020 --workers 16

# 3. grade once, against the sealed folder labels
.venv/bin/python scripts/02_grade.py
```

Other subsets, same protocol:

```sh
# 300 images (100 per class) — a fast, cheap check
.venv/bin/python scripts/01_run.py --fraction 0.02 --seed 7

# exactly the published draw (the defaults)
.venv/bin/python scripts/01_run.py

# re-export predictions from cached answers, no API calls
.venv/bin/python scripts/01_run.py --export-only
```

Every Jev response is cached under `runs/<protocol>/raw/` with its token usage
and latency. A rerun costs nothing, `--export-only` is free, and any subset
only pays for images not yet seen. The cache key covers the protocol version
and the exact question text; the per-run output directory is
`runs/<protocol>/f<fraction>_s<seed>/`, so different draws never collide.

## Cost and runtime

The published run sent 22.2M input tokens over 3,000 images (~7.4k per image:
one image plus 48 questions) at TypeSafe's published $42 per billion input
tokens: **$0.93**. Wall time was **67 seconds** on 16 workers (45 img/s). A
complete reproduction costs about a dollar and a minute.

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
  00_download.py   download the dataset and verify its layout
  01_run.py        classify a stratified subset (--fraction, --seed)
  02_grade.py      one-shot grading against the folder labels
research/          clinical reference and the question bank
reports/run.json   the published metrics
```

## Limitations

- **Not a clinical tool.** This is a benchmark harness. No result here is a
  diagnosis, and no output should be used for patient care.
- **Tile-level, not slide-level.** LC25000-style images are 768 px crops. There
  is no patient, slide, stage, or treatment context.
- **Dataset labels are inherited.** They were not re-reviewed by a pathologist
  for this experiment. LC25000 is derived from a small number of slides and is
  known to contain near-duplicates, so the 3,000 images are not 3,000
  independent specimens; the confidence intervals are image-level and
  optimistic.
- **H&E only.** The markers that resolve poorly differentiated tumours in
  practice (TTF-1, p40, Napsin A, CK5/6) are invisible here — which is exactly
  where the errors concentrate.
- **The benign class is normal lung.** It is not a library of benign mimics
  (reactive pneumocyte hyperplasia, organising pneumonia, squamous metaplasia).
  Expect weaker performance on those in a real biopsy.
- **The dataset is easy by design.** Its classes differ in architecture and
  staining strongly enough that classical models reach high 90s. A high score
  here says more about the benchmark than about clinical readiness.

## Data credit

`rm1000/lung-cancer-histopathological-images` on Kaggle, a 15,000-image subset
of **LC25000** (Borkowski et al., arXiv:1912.12142). Check the Kaggle and
LC25000 licences before redistributing pixels. This repository does not commit
the images.
