# DRAFT — Occlusion-Aware Landmark Recognition by Dynamic Feature Masking

Siddharth Mehta · Northeastern University · mehta.siddh@northeastern.edu

> **How to use this file.** This is draft prose sized for a 6–8 page IEEE two-column paper
> (~5,200 words). Every number has been checked against the run outputs. Edit it into your
> own voice — do not submit it verbatim. Items marked **[FILL]** need something from you.
> Figure and table placement is in `WRITING_GUIDE.md`.

---

## Abstract

Crowdsourced landmark datasets such as Google Landmarks Dataset v2 (GLDv2) are assembled
from tourist photography and therefore contain substantial transient content — people,
vehicles, animals and personal effects — that is unrelated to the landmark being labelled.
We investigate whether explicitly detecting and removing this content improves
instance-level landmark recognition. We build an occlusion-quantification pipeline that
applies YOLOv8m instance segmentation to 80,000 GLDv2 images, filters detections through a
20-class transient taxonomy, and assigns each image a continuous occlusion ratio. Three
otherwise-identical recognition models are trained on raw, fully masked, and stochastically
masked inputs, and each is evaluated on both raw and masked test images to separate genuine
representation improvement from mere train/test distribution alignment. Masking yields a
small positive but statistically indistinguishable change in Global Average Precision
(ΔGAP +0.0079, 95% CI [−0.0010, +0.0163], p = 0.084). However, the cross-condition
evaluation reveals a large, monotone and highly significant asymmetry: removing transient
content at test time degrades the baseline model progressively with occlusion, reaching
−0.4171 GAP in the most occluded stratum (p < 0.001), while a model trained on masked
inputs attains equivalent accuracy without access to those pixels. We conclude that
transient content carries no task-necessary information, that masking is therefore safe but
not beneficial for accuracy, and that its value lies in invariance to occluder statistics
rather than in recognition performance.

**[FILL]** Index Terms — instance-level recognition, landmark recognition, occlusion,
instance segmentation, shortcut learning, Global Average Precision.

---

## I. Introduction

Instance-level landmark recognition differs from generic image classification in a way that
matters for how we should think about data quality. The class is not a category but a
specific physical structure at a specific location. Any visual feature in the image that is
not attached to that structure is, by construction, not evidence for the label. A tour
group, a parked coach, a food cart or a flock of pigeons may appear in many photographs of a
particular monument, but none of them is the monument.

This creates a specific risk. Google Landmarks Dataset v2 [1] is assembled from Wikimedia
Commons, so its images are tourist photographs, and tourist photographs contain tourists. A
model minimising training loss has no reason to distinguish between features of the landmark
and features that merely co-occur with it in the dataset. If a particular monument is
disproportionately photographed in a season when a certain kind of scaffolding is present,
or by visitors carrying a distinctive kind of umbrella, a network can and will use that
signal. This is shortcut learning [2]: the model is not wrong on the training distribution,
but its decision rule depends on something that will not generalise to a deployment
distribution where occluder statistics differ.

Existing work on this benchmark addresses the problem from the model side. The strongest
published systems use angular-margin losses [3], global–local descriptor fusion [4], and
retrieval-based re-ranking. All of these make a model better at *tolerating* irrelevant
content; none of them removes it. Moreover, published results report aggregate Global
Average Precision (GAP) over the whole test set, which means that no existing result tells
us whether reported gains are uniform across occlusion levels or concentrated in the clean
majority of images. That distinction is exactly what an occlusion-focused study needs to
resolve.

We take the complementary approach: intervene on the input. Using a pretrained instance
segmentation model, we identify pixels belonging to semantically transient object classes
and replace them before the recognition model ever sees the image. Our contributions are:

1. **An automated occlusion-quantification pipeline.** Instance segmentation, a curated
   taxonomy of transient object classes, and a continuous per-image occlusion ratio,
   computed once over 80,000 GLDv2 images and released as a reusable index.
2. **A controlled three-arm comparison.** Raw, fully masked and stochastically masked
   training, identical in code, configuration, seed, splits and schedule; the input
   transform is the only difference.
3. **A 2×2 cross-condition evaluation.** Each trained model is evaluated on both raw and
   masked test images. This separates "masking produced better features" from "masking
   merely made training and test distributions match" — a confound that a single
   before/after number cannot detect, and which we show is the dominant effect.
4. **A failure analysis of COCO-pretrained detection on landmark imagery.** We show that an
   off-the-shelf detector labels statues as people and building facades as boats, so that
   naive masking removes the landmark itself, and we develop the taxonomy restrictions,
   per-class thresholds and geometric guard needed to make automated masking safe.

Our headline result is negative and we state it plainly: masking does not significantly
improve GAP, in any occlusion stratum. The result that survives statistical scrutiny is
different and, we argue, more interesting. The baseline model's predictions depend heavily
on transient content — removing it at test time costs 0.4171 GAP in the most occluded
stratum — whereas a model trained without that content achieves the same accuracy without
it. The information is redundant, the dependency is avoidable, and masking buys invariance
rather than accuracy.

---

## II. Related Work

**Google Landmarks and instance-level recognition.** GLDv2 [1] comprises 1,580,470 training
images across 81,313 classes and is the standard benchmark for instance-level recognition
and retrieval. The authors document substantial label noise in the crowdsourced training
split, including images in which the labelled landmark is not visible. Competition solutions
have converged on deep metric learning with angular-margin objectives, most commonly ArcFace
[3], often combined with global–local descriptor fusion such as DOLG [4] and geometric
re-ranking.

**Occlusion robustness.** A substantial literature treats occlusion as something to be
tolerated, through occlusion-aware pooling, part-based representations, or attention
mechanisms that down-weight unreliable regions. A second line injects synthetic occlusion as
regularisation: Random Erasing [5], Cutout [6] and Hide-and-Seek [7] all mask random image
regions during training and consistently improve generalisation. Our work is positioned
between these: we remove *semantically identified* occlusion rather than tolerating it or
injecting it randomly. Our third arm, which applies semantic masking with probability 0.5,
is a direct bridge to the regularisation literature and, as our results show, is the most
practically useful of the three configurations.

**Segmentation as preprocessing.** Using a pretrained detector to preprocess inputs for a
downstream task is well established, but it carries a domain-shift hazard that is rarely
quantified: the detector's training distribution may differ sharply from the target domain.
COCO [8], on which the YOLO family [9] is typically pretrained, contains 80 everyday object
categories and no category for statuary, monuments or architectural detail. Section III-C
shows that this omission is not incidental — it causes the detector to label the very
objects the downstream task cares about, and it must be handled explicitly for masking to be
safe.

**[FILL]** Add citations for shortcut learning [2] and any additional occlusion work your
course expects.

---

## III. Method

### A. Evaluation protocol

The Landmark Recognition 2021 competition test labels were never publicly released and the
challenge is closed; the public `test/` directory contains approximately ten placeholder
images. All metrics in this paper are therefore computed on a held-out split of the labelled
training set. This has two consequences we state up front. First, our absolute GAP values
are not comparable to published leaderboard scores. Second, our test set contains no
non-landmark distractor images, whereas the competition test set does, so our task is
strictly closed-set: every evaluation image has a landmark label.

### B. Subset construction

Training on the full dataset is infeasible within the available compute budget, so we
construct a fixed subset. We rank the 81,313 classes by image count, retain the top 1,000,
and sample at most 80 images from each, giving exactly 80,000 images. Each class is then
split independently into 64 training, 8 validation and 8 test images (an 80/10/10 ratio).

Three properties of this construction are deliberate.

*Frequency ranking enriches the phenomenon under study.* The most photographed landmarks are
also the most heavily visited, and therefore the most occluded by tourists. Sampling classes
uniformly at random would have diluted precisely the effect we set out to measure.

*The image cap eliminates class imbalance as a confound.* Because even the 1,000th-ranked
class has 142 images available (the most frequent has 6,272), every class receives exactly 80, and the per-class
split counts have a standard deviation of zero. Any difference we observe between arms
cannot be attributed to differing class frequency. The cost is that our benchmark does not
exhibit the long tail of the full dataset.

*Splitting within each class* guarantees that every class is represented in training; a
global random split can starve rare classes entirely.

### C. Transient-object detection

#### 1) Detector configuration

We use YOLOv8m-seg at 640 px input resolution, with confidence threshold 0.10, NMS IoU 0.70,
and a cap of 100 detections per image. Two choices warrant explanation.

**Instance segmentation rather than bounding boxes.** A bounding box drawn around a person
standing in front of a cathedral encloses a large rectangle of cathedral. Removing it would
delete substantial landmark evidence along with the occluder. Since the hypothesis under
test is that removing non-landmark pixels helps, an over-inclusive removal operator would
confound the experiment at its root. Polygon masks remove the person and comparatively
little else.

**Detect permissively, filter offline.** Inference stores a superset of detections — every
COCO class except a small never-mask list, down to confidence 0.10 — as normalised polygons
in a columnar file. Taxonomy membership, per-class confidence thresholds and the geometric
guard described below are all applied later, at mask-render time. This is not merely an
optimisation: it means every ablation over those parameters costs seconds of CPU rather than
another GPU pass over 80,000 images, and it makes the detection stage a fixed, auditable
artefact.

The detection cap of 100 is a practical necessity rather than a modelling choice. At the
library default of 300, each detected instance carries a full-resolution mask tensor, and
the resulting memory (300 × 640 × 640 × 4 bytes per image, times the batch) exhausts a 15 GB
GPU before any useful batch size is reached.

#### 2) Transient taxonomy

We define twenty COCO classes as transient, organised in four tiers: *people* (person);
*vehicles* (bicycle, car, motorcycle, bus, truck); *animals* (bird, cat, dog, horse, sheep,
cow, elephant, bear, zebra, giraffe); and *portable objects* (backpack, umbrella, handbag,
suitcase).

Four classes are explicitly excluded from masking under all configurations: **airplane,
train, clock and boat**. Each of these can itself be the landmark — museum aircraft,
preserved locomotives, tower clocks such as Big Ben or the Prague Astronomical Clock, and
historic vessels. We argue this exclusion list is not a quirk of our setup but a general
requirement: any semantic masking scheme for landmark recognition must enumerate the classes
whose instances may constitute the subject, and doing so requires domain knowledge that a
generic object vocabulary does not supply.

#### 3) Detector failure on landmark imagery

An initial configuration using YOLOv8s-seg with a uniform confidence threshold of 0.25 and
no geometric filtering produced systematic and damaging false positives. On a 200-image
diagnostic sample we observed 48 `boat` detections, 7 `giraffe` and 1 `elephant` — object
classes with no plausible presence in the corresponding photographs. Four failure modes were
consequential:

| Image content | Detected as | Consequence |
|---|---|---|
| Building facade | boat, kite | Landmark masked out entirely |
| Stone statue | person | Landmark masked out |
| Fireworks display | bird, car, person | Subject masked out |
| Rock formation | (spurious) | Subject partly masked |

The statue case is the most instructive. COCO contains no `statue` category, and carved
human figures fall squarely within the learned `person` distribution. This is a categorical
limitation of transfer from a general object vocabulary, not a threshold that can be tuned
away. Fig. 2 shows all four cases.

We introduced three mitigations.

**Per-class confidence thresholds.** Measured confidence distributions showed medians
between 0.16 and 0.33 across classes with maxima near 0.95, confirming that the false
positives were concentrated in the low-confidence tail. We set the threshold for `person` at
0.30 — the class detectors are most reliable on, and the occluder that matters most — 0.40
for vehicles and `umbrella`, and 0.55 for all other classes.

**Taxonomy narrowing.** The wider COCO sports and tableware classes (frisbee, kite, sports
ball, bottle, wine glass and others) were observed firing on architectural features and were
moved to an opt-in tier, excluded by default. `boat` was moved to the never-mask list.

**Dominant-subject guard.** We reject any detection covering at least 25% of the frame whose
centre lies within 0.30 (normalised) of the image centre. The principle is that an occluder
is not the photographic subject: a person the photographer intended to exclude is not
usually large and centred. The guard is defensible rather than arbitrary because the median
detection area in our data is 0.002 of the frame, two orders of magnitude below the
threshold, so it fires only on the extreme tail. We note the cost explicitly: a genuinely
large, centred foreground occluder is preserved rather than removed. The guard is a
configuration flag and can be ablated.

Together these reduced the occlusion ratio on the fireworks image from 0.36 to 0.00, on the
rock formation from 0.13 to 0.00, and on the statue from 0.39 to 0.10. Residual false
positives remain — small patches persist on the Lotus Temple dome and the Sydney Opera House
roof — so the mitigation is partial rather than complete.

#### 4) Occlusion ratio

For each image we render the union of retained transient regions, dilated by 4 px to capture
boundary pixels, and define the **occlusion ratio** as the fraction of image area covered.
This is a proxy derived from an imperfect detector, not a ground-truth measurement, and we
treat it as such throughout.

Across the 80,000-image subset the mean ratio is 0.0189, the median is 0.000, the maximum is
0.816, and **67.0% of images contain no detected transient content at all**. We partition
images into four strata at fixed cut points of 0.02, 0.10 and 0.25:

| Stratum | Range | All 80k | % | Test |
|---|---|---|---|---|
| none | < 0.02 | 65,115 | 81.4 | 6,521 |
| low | 0.02 – 0.10 | 10,436 | 13.0 | 1,039 |
| medium | 0.10 – 0.25 | 3,387 | 4.2 | 317 |
| high | ≥ 0.25 | 1,062 | 1.3 | 123 |

Stratum proportions are near-identical across the train, validation and test splits,
confirming that stratified splitting did not induce a correlation between split membership
and occlusion. Binning is applied at analysis time to a stored continuous value, so no model
selection depends on the cut points.

### D. Masking strategies

We implement four fill operations — constant black, per-image mean fill, Gaussian blur and
Telea inpainting — and use mean fill throughout. Constant black is a poor default despite
being the obvious choice: a solid black region is itself a strong, spatially coherent and
learnable signal, so a network can key on mask *shape* rather than ignoring the masked
region, which would inflate the masked arm's score for a reason unrelated to the hypothesis.
Mean fill computes the mean colour of the **retained** pixels only; including masked pixels
would bias the fill toward the very occluder being removed.

Masking is skipped entirely when it would cover more than 85% of an image. We deliberately
kept this ceiling high. Images above roughly 0.5 occlusion are, on inspection, indoor group
photographs in which no landmark is visible — GLDv2 label noise that no masking strategy can
repair. Lowering the ceiling would have left the `high` stratum effectively unmasked in the
masked arm, making the comparison vacuous at precisely the occlusion level the hypothesis
concerns.

Masking is applied to the decoded image *before* augmentation, since it models a cleaning
operation on the source photograph rather than a geometric transform.

### E. Experimental arms

| Arm | Training input | Role |
|---|---|---|
| `baseline` | raw | control |
| `masked` | transient regions removed, p = 1.0 | treatment |
| `maskaug` | masked with p = 0.5 | treatment as stochastic augmentation |

All three arms share identical code, configuration, random seed (42), data splits, optimiser
schedule and augmentation pipeline. Masking is a runtime argument to the dataset rather than
a property of the arm, which is what makes the cross-condition evaluation of Section V-B
mechanically possible: any trained model can be evaluated under any input condition.

---

## IV. Experimental Setup

**Model.** An EfficientNet-B0 backbone (timm identifier
`efficientnet_b0.ra4_e3600_r224_in1k`, 5.18 M parameters) produces pooled features, followed
by a neck of BatchNorm, dropout (0.2), a linear projection to a 512-dimensional embedding,
and a second BatchNorm. The classification head is ArcFace with scale 30 and angular margin
0.30. ArcFace is appropriate here because the task is fine-grained and instance-level:
angular margin separates 1,000 visually similar classes more effectively than plain softmax,
and it is standard in strong landmark systems. A plain linear head is implemented as a
control but not used in the reported experiments.

**Training.** Inputs are 224 × 224 crops drawn from a 256 px short-side cache, with
RandomResizedCrop (scale 0.7–1.0), horizontal flipping and colour jitter. We use AdamW with
learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, a cosine schedule with one epoch of linear
warmup, batch size 64, mixed precision, and gradient clipping at 5.0, for 15 epochs.
Checkpoints are selected by best validation GAP with an early-stopping patience of 5 epochs.

**Metric.** Global Average Precision is defined as

  GAP = (1/M) Σᵢ P(i) · rel(i)

where predictions across all images are ranked by confidence, P(i) is precision at rank i,
rel(i) indicates whether prediction i is correct, and M is the number of images containing a
landmark. The ranking is global rather than per-image, which means a confidently wrong
prediction degrades precision for every lower-ranked prediction and is therefore penalised
more heavily than an uncertain one. Confidence calibration is thus part of the metric — which
is desirable here, since removing occluders should plausibly affect a model's certainty as
well as its argmax. In our closed-set setting M equals the number of evaluation images.
Stratified GAP is computed within each stratum, so strata are mutually comparable but do not
decompose the overall figure.

We select checkpoints on validation GAP rather than accuracy so that model selection
optimises the quantity we report.

**Statistical testing.** We report paired bootstrap confidence intervals over 1,000
resamples of the evaluation images, applying identical resampling indices to both arms.
Pairing is appropriate because both arms are evaluated on the same images, and it removes
between-image difficulty as a source of variance. We additionally report McNemar's test on
per-image top-1 correctness. The two tests probe different quantities — ranking and accuracy
respectively — so agreement between them is more informative than either alone.

**Implementation.** Detection and image caching are fused into a single GPU pass over the
source data, resizing directly from the array the detector returns and avoiding a second
read of the 98 GB dataset. The pass takes approximately 75 minutes on a single NVIDIA T4 and
produces a 2.40 GB image cache, 32.8 MB of binary masks, and a 224 MB detection record.
Training costs 310 s per epoch for the baseline arm, 390 s with full masking (+25.8%) and
355 s with stochastic masking (+14.5%).

We note one implementation detail with methodological weight. Our cache-building stage
asserts that the in-memory manifest agrees with the files written to disk. This assertion
caught a silent identifier collision in which the detection library returned generic
sequential names for list-valued inputs, causing an entire cache to be written under four
filenames rather than several thousand. The failure produced no error and would have yielded
plausible-looking downstream metrics. We record it because cache-building pipelines of this
kind require manifest–filesystem cross-validation as a matter of course.

---

## V. Results and Analysis

All results are computed on the held-out test split of 8,000 images across 1,000 classes.
Checkpoints were selected on validation GAP, so the test split is uncontaminated by model
selection. One training run per arm, seed 42 throughout.

### A. Training dynamics

Fig. 5 shows validation GAP per epoch for the three arms. Final training losses are nearly
identical (0.28, 0.29 and 0.30 for baseline, masked and maskaug respectively, from
comparable starting values near 13.5), which serves as a negative control: masking neither
made the task harder to fit nor removed the signal needed to fit it. The masked arm peaked
at epoch 12 and drifted slightly downward thereafter, while the other two were still
improving at epoch 14 — consistent with masked inputs offering less variety, so that the
model saturates sooner.

Best validation GAP was 0.7662 (baseline), 0.7507 (masked) and 0.7603 (maskaug). These
figures are **not** comparable to one another, because each arm validates on its own input
condition; the comparison is made in Section V-B.

### B. Cross-condition evaluation

Table 3 reports test GAP for every combination of training arm and evaluation input.

| train ↓ / eval → | raw | masked | spread |
|---|---|---|---|
| baseline | 0.7588 | 0.7443 | −0.0145 |
| masked | 0.7623 | **0.7667** | +0.0044 |
| maskaug | 0.7646 | 0.7641 | −0.0005 |

Two observations follow. First, the masked-trained model scores 0.7623 on *raw* images,
above the baseline's 0.7588 on the same images — that is, it performs better on an input
condition it never saw during training. This argues against the interpretation that any
benefit of masking is simply train/test distribution alignment, which is the confound the
cross-condition design exists to detect. The margin is small (+0.0035) and the confidence
intervals overlap substantially, so we do not rest any claim on it alone.

Second, and more strikingly, the arms differ sharply in how sensitive they are to the input
condition. Removing occluders at test time costs the baseline 0.0145 GAP, whereas the masked
arm gains slightly and maskaug is flat to within 0.0005. This asymmetry is the largest
effect anywhere in our results and is developed in Section V-D.

### C. The stated hypothesis is not supported

Table 4 reports matched-condition GAP by occlusion stratum with paired bootstrap confidence
intervals.

| stratum | n | baseline/raw | masked/masked | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| none | 6,521 | 0.7524 | 0.7589 | +0.0065 | [−0.0034, +0.0158] | 0.215 |
| low | 1,039 | 0.8075 | 0.8245 | +0.0170 | [−0.0027, +0.0370] | 0.098 |
| medium | 317 | 0.7689 | 0.7733 | +0.0044 | [−0.0411, +0.0461] | 0.848 |
| high | 123 | 0.6475 | 0.6587 | +0.0112 | [−0.0752, +0.0963] | 0.764 |
| **ALL** | 8,000 | 0.7588 | 0.7667 | +0.0079 | [−0.0010, +0.0163] | 0.084 |

The differences are positive in every stratum, but none is statistically significant, and
they show no monotone relationship with occlusion. The overall difference of +0.0079 has a
confidence interval that includes zero (p = 0.084); McNemar's test is likewise
non-significant, with the masked arm correcting 548 baseline errors while introducing 508 of
its own (p = 0.230). The stochastically masked arm behaves similarly, whether evaluated on raw input
(ΔGAP +0.0058, CI [−0.0027, +0.0133], p = 0.189) or masked input (ΔGAP +0.0053,
CI [−0.0031, +0.0133], p = 0.236).

We therefore find no support for the hypothesis as originally stated. The direction of the
effect is consistent with it, but the magnitude is not resolvable at this sample size with a
single training run per arm.

One structural fact explains why an overall effect was always unlikely to be large: 81.4% of
test images fall in the `none` stratum, where masking is by construction a no-op. Any
aggregate metric is dominated by images the intervention cannot affect.

### D. A monotone asymmetric dependency on transient content

Table 5 reports what happens when a model is evaluated on the input condition it was *not*
trained on.

| stratum | n | baseline: raw → masked | 95% CI | p |
|---|---|---|---|---|
| none | 6,521 | +0.0006 | [−0.0006, +0.0017] | 0.369 |
| low | 1,039 | −0.0249 | [−0.0374, −0.0140] | < 0.001 |
| medium | 317 | −0.1393 | [−0.1858, −0.0967] | < 0.001 |
| high | 123 | **−0.4171** | [−0.5175, −0.3155] | < 0.001 |

The baseline degrades monotonically with occlusion when transient content is removed,
falling from 0.6475 to 0.2304 in the `high` stratum — a 64% relative collapse, roughly four
times the confidence-interval half-width. The reverse mismatch is far milder: the masked
model loses 0.1174 GAP on raw input in the same stratum (p = 0.007), about a third as much.
The stochastically masked arm is nearly condition-invariant, with a worst-stratum change of
−0.0429 that is not statistically significant. Fig. 6 shows the divergence.

An obvious objection is that this is distribution shift rather than dependency: in the
`high` stratum masking blanks 36.5% of the image on average, and any model would degrade on
inputs so far outside its training distribution. Our design answers this directly. The
masked model attains 0.6587 in the `high` stratum **without access to any transient
pixels**, statistically indistinguishable from the baseline's 0.6475 *with* them
(Δ +0.0112, CI [−0.0752, +0.0963]). If those pixels carried information necessary for the
task, no model trained without them could match a model trained with them. They do not.

### E. Internal control

The `none` stratum provides a control for the masking operation itself. Comparing
`baseline/masked` against `baseline/raw` on those 6,521 images yields Δ = +0.0006,
CI [−0.0006, +0.0017], p = 0.369. Where there is nothing to mask, masking changes nothing.

This rules out the most serious alternative explanation for Table 5 — that the mean-fill
artefact perturbs predictions independently of what it covers. Without this control, the
degradation reported above could not be attributed to the removal of transient content
specifically.

### F. Qualitative analysis

**[FILL]** Fig. 7 shows examples drawn from the per-image prediction records. The upper row
shows images the baseline classifies correctly and the masked arm does not, illustrating the
cost of over-masking documented in Section III-C-3. The lower row shows the converse.
McNemar's discordant counts (548 corrections against 508 regressions) indicate that this
trade is close to even, and we present both directions accordingly.

*Write two or three sentences describing what you actually see in your chosen examples —
whether the regressions cluster on a particular kind of image, e.g. statues or crowded
plazas.*

### G. Synthesis

Taken together, our results support a three-part conclusion. Masking is **sufficient**: the
transient content it removes carries no information necessary for landmark recognition, as
demonstrated by the equivalence at high occlusion in Section V-D. It is **safe**: we measure
no accuracy cost in any stratum. But it is **not beneficial**: we measure no accuracy gain
either. Its value is invariance rather than accuracy.

This has a practical implication. A model trained on raw crowdsourced data acquires a
dependency on occluder content that is invisible under standard evaluation, because
evaluation data shares the occluder statistics of the training data. Deployed where those
statistics differ — a different season, a different visitor demographic, a different vehicle
fleet — the dependency becomes a failure mode. Masking removes it at no measured accuracy
cost. Stochastic masking achieves nearly the same invariance at 14.5% additional per-epoch
cost rather than 25.8%, and is our practical recommendation.

---

## VI. Discussion and Limitations

**Scale.** Our benchmark comprises 1,000 balanced classes and 80,000 images, against the
full dataset's 81,313 classes and 1.58 M images with a heavy tail. Results may not transfer
to that regime.

**Protocol.** We evaluate on a held-out training split rather than the competition test set,
and our test data contains no non-landmark distractors. Absolute GAP is therefore not
comparable to published leaderboard figures.

**The occlusion ratio is a proxy.** It is derived from the same detector whose failures we
document in Section III-C-3, which introduces a circularity: images on which the detector
fails are also the images we mis-stratify. Validating the proxy would require human
annotation of a subsample.

**Categorical limitation of the detector.** COCO has no statuary category. Our
dominant-subject guard mitigates the resulting failures but does not solve them, and it
introduces its own cost by preserving large centred occluders.

**Single seed per arm.** We have no estimate of run-to-run variance, so differences smaller
than seed noise cannot be resolved. Three to five seeds per arm would be required; this was
beyond the available GPU budget.

**Statistical power, quantified.** The `high` stratum confidence interval has a half-width
of ±0.095 at n = 123. Resolving the observed +0.011 effect would require an interval roughly
8.6 times tighter, hence about 9,100 high-occlusion test images — at the measured 1.54%
prevalence, a test set of approximately 593,000 images. The study is thus underpowered not
by design error but by the rarity of the phenomenon, and no experiment at this scale could
resolve an effect this small. This directly motivates the occlusion-enriched sampling
proposed in Section VII.

**Occlusion is not randomly assigned.** The `low` stratum outperforms `none` in every cell
of our results (0.8075 against 0.7524 for the baseline on raw input). Lightly occluded images
are *easier* than clean ones, most plausibly because photographs containing people tend to be
canonical, well-framed shots of famous landmarks, whereas the `none` stratum includes
interiors, detail crops and atypical viewpoints. Comparisons *across* strata are therefore
confounded; only the within-stratum comparisons we report are clean.

**Ambiguity in the maskaug condition.** Having trained with masking probability 0.5, neither
raw nor masked input is unambiguously its matched condition. We report both.

**Fill strategy.** Only mean fill was evaluated at full scale. Our conclusions are properly
read as applying to mean-fill masking rather than to masking in general.

**Label noise at the extremes.** Images above roughly 0.5 occlusion are frequently group
photographs with no visible landmark. Masking cannot repair a mislabelled image, and such
images dilute the `high` stratum.

---

## VII. Future Work

**Immediate ablations enabled by the stored detections.** Because detection results are
stored as a filterable superset, several ablations require no further GPU inference:
comparing black, blur and inpainting fills against mean fill; adding or removing taxonomy
tiers; and quantifying the precision/recall trade of the dominant-subject guard. Repeating
each arm across three to five seeds would establish whether the consistent positive
direction we observe survives seed variance.

**Feature-space masking.** The most promising methodological extension is to stop
overwriting pixels. Pixel-space fills inject an artefact that the network can itself learn.
Suppressing the corresponding positions in the backbone's feature map, or supplying the mask
as an additional input channel so the network is *told* which regions are unreliable rather
than handed a fabricated fill, would remove that artefact while preserving the intervention.

**Occlusion-aware training objectives.** Rather than modifying inputs, one could down-weight
heavily occluded samples in the loss, or order training as a curriculum from clean to
occluded images.

**Better occlusion estimation.** Fine-tuning a segmentation model on landmark imagery with
an explicit statuary or monument class would address our worst failure mode directly.
Open-vocabulary detection would allow transient classes to be specified by description
rather than by a fixed 80-class vocabulary. Monocular depth estimation offers an orthogonal
signal, separating foreground occluders from background structure geometrically rather than
semantically.

**Occlusion-enriched evaluation.** Our power analysis shows that resolving effects of the
magnitude we observe requires far more high-occlusion images than random sampling provides.
A test set stratified to over-sample occluded images — with results reweighted to the natural
prevalence — would resolve the same effect with a fraction of the annotation and compute.

**Scale and protocol.** Extending to the full 81,313-class problem, larger backbones and
higher input resolution, and adding non-landmark distractors to match the competition
protocol, would all bring the setting closer to the published benchmark.

---

## VIII. Conclusion

We asked whether explicitly detecting and removing transient content improves landmark
recognition on crowdsourced data. Across 80,000 GLDv2 images and three controlled training
arms, the answer is that it does not: the change in Global Average Precision is positive in
every occlusion stratum but statistically indistinguishable from zero. What the
cross-condition design revealed instead is a large, monotone and highly significant
dependency in the *baseline* model on content that is not the landmark — a dependency that
costs 0.4171 GAP when that content is removed at high occlusion, and which a model trained
without it does not exhibit while matching its accuracy. Transient content in this dataset
is therefore redundant rather than harmful, and semantic masking should be understood as a
tool for robustness and invariance rather than as a route to higher recognition accuracy. We
also show that off-the-shelf detection cannot be applied naively to this task, since a
COCO-pretrained model removes statues and building facades along with tourists, and we give
the taxonomy restrictions and geometric constraints that make the operation safe.

---

## References

**[FILL]** Format in IEEE style. At minimum:

1. T. Weyand, A. Araujo, B. Cao, and J. Sim, "Google Landmarks Dataset v2 — A Large-Scale
   Benchmark for Instance-Level Recognition and Retrieval," in *Proc. CVPR*, 2020.
2. R. Geirhos et al., "Shortcut Learning in Deep Neural Networks," *Nature Machine
   Intelligence*, vol. 2, 2020.
3. J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep
   Face Recognition," in *Proc. CVPR*, 2019.
4. M. Yang et al., "DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local
   and Global Features," in *Proc. ICCV*, 2021.
5. Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in
   *Proc. AAAI*, 2020.
6. T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural Networks
   with Cutout," *arXiv:1708.04552*, 2017.
7. K. K. Singh and Y. J. Lee, "Hide-and-Seek: Forcing a Network to be Meticulous for
   Weakly-Supervised Object and Action Localization," in *Proc. ICCV*, 2017.
8. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in *Proc. ECCV*, 2014.
9. G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online].
   Available: https://github.com/ultralytics/ultralytics
10. Kaggle, "Google Landmark Recognition 2021." [Online].
    Available: https://www.kaggle.com/competitions/landmark-recognition-2021

**Reproducibility.** Code, frozen split manifests (`subset_splits.csv`, `class_map.csv`,
`occlusion_index.csv`) and per-image prediction records are available at **[FILL: repo URL]**.
All experiments use seed 42.
