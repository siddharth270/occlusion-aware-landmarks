# Writing Guide — IEEE Two-Column Paper

**Occlusion-Aware Landmark Recognition by Dynamic Feature Masking**
Siddharth Mehta · mehta.siddh@northeastern.edu

> This is a **guide**, not the paper. It tells you what each section must contain,
> gives you every number you need, and names the argument to make. Write the prose
> yourself. Keep `report.md` (or your `.tex`) for the actual paper.

**Target structure** for ~6–8 pages in `IEEEtran`:
Abstract · I. Introduction · II. Related Work · III. Method · IV. Experimental Setup ·
V. Results and Analysis · VI. Discussion & Limitations · VII. Future Work ·
Conclusion · References

---

## I. Introduction

Four paragraphs, in this order.

### ¶1 — The problem, framed as a *data* problem, not a model problem

Instance-level landmark recognition differs from generic classification: the class is a
**specific physical structure**, so any image feature not attached to that structure is
by definition noise. GLDv2 is crowdsourced from Wikimedia, so images are tourist
photographs — and tourist photographs contain tourists.

State the concern precisely: a model can achieve low training loss by keeping features
that correlate with a landmark *in the dataset* but not with the landmark *in the world*
— a particular tour-group uniform, a vendor cart, a bus livery. That is a
**shortcut-learning** framing, and it is the right one to invoke.

### ¶2 — The gap in existing work

Competition solutions attack this with architecture and loss design (ArcFace variants,
DOLG, global–local fusion, re-ranking) — all of which make the model better at
*tolerating* noise. Almost nothing intervenes on the input to *remove* it.

And critically: published landmark results report aggregate GAP, so nobody reports
whether the reported gains are uniform across occlusion levels or concentrated in the
clean majority. **Name that as the gap.**

### ¶3 — Your contribution, as an enumerated list

1. An automated occlusion-quantification pipeline: pretrained instance segmentation → a
   taxonomy of transient classes → a continuous per-image occlusion ratio, computed once
   over 80,000 GLDv2 images.
2. A controlled three-arm comparison (raw / masked / stochastically-masked), identical in
   every respect except the input transform.
3. A **2×2 cross-condition** evaluation that separates *"masking produces better features"*
   from *"masking merely makes train and test distributions match"* — a confound a single
   before/after number cannot distinguish.
4. A failure analysis of COCO-pretrained detectors on landmark imagery, showing they
   mislabel **the landmark itself** (statues→person, facades→boat), plus the mitigations
   that make automated masking safe.

### ¶4 — Signpost the finding, then the roadmap

State your finding honestly in one sentence, whatever it turns out to be. **Do not
oversell.** If masking does not win, say so here: a stated negative result read as
deliberate is far stronger than one that emerges apologetically in Section V.

> **Framing tip.** Your strongest single sentence for the abstract and intro is that the
> occlusion distribution is severely zero-inflated — **67.0% of images contain no detected
> transient content**. That means aggregate GAP is dominated by images masking cannot
> affect, which is *itself* the argument for stratified reporting. Lead with it.

---

## II. Related Work

Four short paragraphs, ~½ column.

| ¶ | Topic | The point to make |
|---|-------|-------------------|
| 1 | **GLDv2 & the benchmark** | Cite Weyand et al. CVPR'20. Give scale (1,580,470 images / 81,313 classes). Note the documented label noise in the crowdsourced train split. |
| 2 | **Instance-level recognition methods** | ArcFace/margin losses, global–local descriptors, retrieval re-ranking. One sentence each. The point: all are *model-side*. |
| 3 | **Occlusion robustness** | Occlusion-aware pooling, Random Erasing / CutOut / Hide-and-Seek. Make the distinction explicit: those **inject** random occlusion as regularisation; you **remove** semantically-identified occlusion. Your `maskaug` arm bridges the two literatures — flag it here. |
| 4 | **Detection/segmentation as preprocessing** | YOLO family, and the domain-shift problem when a COCO-trained detector meets out-of-distribution imagery. This sets up §III-C so your failure analysis does not arrive unmotivated. |

---

## III. Method

Write this as the pipeline in order. **Include the revisions inline as design rationale,
not as a confession** — *"X was chosen because Y failed"* is a methods sentence;
*"we initially got it wrong"* is not.

### A. Evaluation protocol — put this first, it constrains everything

- Competition test labels were never released and the challenge is closed; the public
  `test/` directory holds ~10 placeholder images.
- Therefore all metrics come from a **held-out split of the labelled train set**.

State this plainly and early. It is a limitation, but concealing it would be worse, and
stating it up front makes the rest of the design legible.

### B. Subset construction

| Parameter | Value |
|---|---|
| Source | 1,580,470 images / 81,313 classes |
| Classes retained | top **1,000** by image count |
| Images per class | capped at **80** |
| Total | **80,000** |
| Split (per class, stratified) | 64 / 8 / 8 → **64,000 / 8,000 / 8,000** |

Three justifications to state:

- **Compute.** Full scale is infeasible in a 12 h / 30 h-per-week GPU budget.
- **Enrichment, not convenience.** Frequency-ranked selection is deliberate: the
  most-photographed landmarks are the most tourist-occluded, so this *enriches* the
  phenomenon under study. Contrast with random class sampling, which would dilute it.
- **Confound removal.** Because the rank-1000 class still had ~200 images available,
  every class receives exactly 80, giving a perfectly balanced set (per-class split
  counts have standard deviation **0.0**). Class imbalance is therefore eliminated as a
  competing explanation for any observed difference.
  **Cost to acknowledge:** your GAP is not comparable to leaderboard GAP, which faces a
  long tail.
- Splitting **within** each class, not globally, guarantees every class appears in train.

### C. Transient-object detection — *your most original section, give it the most space*

#### C.1 Detector

YOLOv8m-seg, `imgsz=640`, `conf=0.10`, `iou=0.70`, `max_det=100`.

Two choices to defend:

**Segmentation, not bounding boxes.** A box around a person in front of a cathedral
deletes a large rectangle of cathedral. Polygon masks delete the person. Since the entire
hypothesis is that removing non-landmark pixels helps, over-removing landmark pixels would
confound the result. State it that way.

**Detect at low confidence, filter offline.** Inference stores a superset (every COCO class
except the never-mask set, down to conf 0.10) into a parquet with normalised polygons.
Confidence thresholds, taxonomy membership, and the subject guard are all applied at
*mask-render* time. Consequence worth stating: every ablation over those parameters costs
**CPU seconds instead of another GPU pass over 80,000 images**. This is a reproducibility
contribution, not just an optimisation.

#### C.2 Transient taxonomy — 20 COCO classes in four tiers

| Tier | Classes |
|---|---|
| people | person |
| vehicles | bicycle, car, motorcycle, bus, truck |
| animals | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| portable objects | backpack, umbrella, handbag, suitcase |
| **never masked** | airplane, train, clock, boat |

The never-mask set is the interesting part: each of those classes **can be the landmark** —
museum aircraft, preserved locomotives, Big Ben, historic ships. Argue that any semantic
masking scheme needs such a list, and that *requiring* it is a general finding about the
approach, not a quirk of yours.

#### C.3 Failure analysis and mitigation — *the subsection that makes the paper*

Report the diagnostic honestly: on a 200-image sample, YOLOv8s-seg at a uniform 0.25
threshold produced **48 `boat` detections, 7 `giraffe`, 1 `elephant`** in landmark
photographs.

Present the four concrete failures alongside the figure:

| Failure | Detected as | Consequence |
|---|---|---|
| Building facade | boat, kite | Landmark masked out entirely |
| Stone statue | person | Landmark masked out (the statue *is* the landmark) |
| Fireworks display | bird, car, person | Subject masked out |
| Rock formation | (spurious) | Subject partly masked |

**Diagnose the cause:** COCO contains no `statue` class, and stone figures fall within the
learned `person` distribution. That is a **categorical limitation of transfer**, not a
threshold to tune.

Then the three mitigations, each with its rationale:

1. **Per-class confidence thresholds** — `person` 0.30 (the class detectors are best at and
   the occluder that matters most), vehicles and `umbrella` 0.40, everything else **0.55**.
   Justify from the measured confidence distribution: median confidence sat at **0.16–0.33**
   across classes with maxima near **0.95** — the false positives lived in the
   low-confidence tail.
2. **Taxonomy narrowing** — the COCO sports/tableware classes (frisbee, kite, sports ball,
   bottle, wine glass, …) fired on architectural features and were moved to an opt-in tier,
   excluded by default. `boat` moved to never-mask.
3. **Dominant-subject guard** — reject any detection covering ≥ 25% of the frame whose
   centre lies within 0.30 of the image centre. Principle: **an occluder is not the
   photographic subject.**
   The empirical justification that makes it defensible rather than ad hoc: **median
   detection area is 0.002 of the frame**, two orders of magnitude below the 0.25
   threshold, so the guard fires only on the extreme tail.
   Be explicit about the trade: a genuinely large centred foreground occluder is preserved.
   It is a flag, and it is ablatable.

**Quantify the effect** (your before/after figure):

| Case | permissive → strict |
|---|---|
| Fireworks | 0.36 → **0.00** |
| Rock formation | 0.13 → **0.00** |
| Statue | 0.39 → **0.10** |

**Acknowledge the residual:** small false-positive patches persist on the Lotus Temple dome
and the Sydney Opera House.

#### C.4 Occlusion ratio

Define it: union area of retained transient regions (dilated 4 px) divided by image area,
in [0, 1]. It is a **proxy, not ground truth** — say so.

Report: mean **0.0189**, median **0.000**, max **0.816**, and **67.0% exactly zero**.

| Stratum | Range | All 80k | % | Test split |
|---|---|---|---|---|
| none | < 0.02 | 65,115 | 81.4 | 6,521 |
| low | 0.02 – 0.10 | 10,436 | 13.0 | 1,039 |
| medium | 0.10 – 0.25 | 3,387 | 4.2 | 317 |
| high | ≥ 0.25 | 1,062 | 1.3 | **123** |

Bin edges are fixed a priori at 0.02 / 0.10 / 0.25 and applied at analysis time to a
stored continuous value, so **no model selection depends on them**. Note the consequence
to report: the `high` stratum holds only 123 test images, so per-stratum GAP there carries
wide confidence intervals and must never be reported as a bare point estimate.

### D. Masking strategies

Four fills implemented: `black`, `mean_fill`, `blur`, `inpaint_telea`. `mean_fill` is the
default.

**Give the reason black is not the default** — it is a good methodological point: a solid
black region is itself a strong, learnable signal, so a network can key on mask *shape*
instead of ignoring the region, which would inflate the masked arm's score for the wrong
reason.

`mean_fill` uses the mean of the **retained** pixels only; including masked pixels would
bias the fill toward the occluder being removed.

`max_mask_fraction = 0.85` skips masking that would erase nearly the whole image. Report
the observation that motivated keeping it *high* rather than lowering it: images above
~0.5 occlusion are indoor group photographs where the landmark is absent — GLDv2 label
noise. Lowering the cap would leave the `high` stratum **unmasked** in the masked arm,
making the comparison vacuous at exactly the occlusion level the hypothesis concerns.
**That reasoning belongs in the paper** — it shows you understood the trade rather than
picked a number.

### E. Experimental arms

| Arm | Training input | Purpose |
|---|---|---|
| `baseline` | raw | control |
| `masked` | transient regions removed, p = 1.0 | treatment |
| `maskaug` | masked with p = 0.5 | treatment as stochastic augmentation |

State the controlled-experiment claim explicitly: identical code, config, seed (42),
splits, schedule, and augmentation; the sole difference is the input transform.

Note that `apply_masking` is a **runtime argument** to the dataset rather than a property
of the arm — that is what makes the 2×2 evaluation mechanically possible.

---

## IV. Experimental Setup

| | |
|---|---|
| Backbone | `efficientnet_b0.ra4_e3600_r224_in1k` (timm 1.0.26), 5.18 M params |
| Embedding | 512-d; neck BN → dropout(0.2) → linear → BN |
| Head | ArcFace, scale 30, margin 0.30 |
| Input | 224×224, from a 256px-short-side cache |
| Augmentation | RandomResizedCrop(0.7–1.0), hflip, ColorJitter(0.2/0.2/0.2/0.02) |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | cosine, 1 epoch linear warmup |
| Epochs / batch | 15 / 64, mixed precision, grad clip 5.0 |
| Selection | best **validation GAP**, early stop patience 5 |
| Hardware | Kaggle NVIDIA T4, ~5.1 min/epoch |

Two things to justify:

- **Why ArcFace.** Fine-grained instance-level task with 1,000 classes; angular margin
  separates instances better than plain softmax, and it is what strong landmark systems
  use. Mention the `linear` head exists as a control.
- **Why select on GAP, not accuracy.** GAP is the reported metric and it rewards calibrated
  confidence; selecting on accuracy would optimise something the study does not measure.

### Metric definition

$$\text{GAP} = \frac{1}{M}\sum_{i=1}^{N} P(i)\cdot \text{rel}(i)$$

Predictions are ranked by confidence **globally across all images**. The consequence to
spell out: a confidently wrong prediction pollutes precision for every lower-ranked
prediction, so it is punished harder than an uncertainly wrong one. **Confidence
calibration is therefore part of the metric** — which is exactly the quantity that should
change when occluders are removed.

You can cite your own verification: on a synthetic 3-image example, one confident error
scores **0.389** versus **0.667** for the same error made uncertainly.

Note that $M = N$ here (closed set, every image labelled), and that stratified GAP is
computed within each stratum ($M$ = stratum size), so strata are mutually comparable but
**do not decompose** the overall figure.

### Statistical testing

- **Paired bootstrap** (1,000 resamples of images, same indices applied to both arms) for
  the GAP difference and its 95% CI.
- **Exact-binomial McNemar** on per-image top-1 correctness.

Justify pairing: identical test images, so pairing removes between-image difficulty as a
variance source. Justify reporting both: the bootstrap tests **ranking**, McNemar tests
**accuracy**; agreement between two different tests is more persuasive than either alone.

### Engineering notes — worth one short paragraph

Reviewers of a systems-flavoured paper value these, and they are real work.

- Detection and caching were **fused into one pass** using the detector's returned decoded
  image, avoiding a second read of a 98 GB mounted dataset.
- `max_det` had to be capped at **100**: at the default 300, per-instance mask tensors
  (300 × 640 × 640 × 4 bytes × batch) exhaust a 15 GB T4.
- An **integrity assertion** comparing the in-memory manifest against files on disk caught
  a silent identifier collision in which the detection library returned generic names
  (`image0`, `image1`, …) for list-valued sources, so an entire 2,000-image cache had been
  written under **4 filenames**.
  Frame this as: *cache-building pipelines need manifest–filesystem cross-validation,
  because the failure is silent and downstream metrics would still have looked plausible.*

---
## V. Results and Analysis

> All numbers below are final, from the held-out **test** split (8,000 images, 1,000
> classes). Checkpoints were selected on validation GAP, so the test split is uncontaminated
> by model selection. Seed 42, one run per arm.

**Write §V in this order.** The narrative is: *the stated hypothesis fails → but a larger,
cleaner effect appears in the off-diagonal → and here is the control proving it is real.*
Leading with the null is deliberate — it buys you credibility for Finding 2.

### A. Training dynamics

**Figure:** `training_curves.png` — val GAP vs epoch, three arms.

| Arm | Best val GAP | Epoch | Val condition | s/epoch |
|---|---|---|---|---|
| baseline | 0.7662 | 14 | raw | 310 |
| masked | 0.7507 | 12 | masked | 390 |
| maskaug | 0.7603 | 14 | masked | 355 |

Three sentences only:
- **Training loss curves are nearly identical** across arms (13.5 / 13.6 / 13.5 → 0.28 /
  0.29 / 0.30). Masking neither made the task harder to fit nor destroyed learnable signal.
  This is a **negative control**: it rules out "the masked arm simply had less to learn from."
- The masked arm **peaked at epoch 12 and drifted down**, while baseline and maskaug were
  still rising at 14 — consistent with masked inputs carrying less variety, so the model
  saturates sooner. Selection on val GAP handled it.
- **Warn the reader** that these three numbers are *not* comparable: each arm validated on
  its own input condition. The comparison is §V-B.

### B. Cross-condition matrix

**Table 3.** Test-set GAP, rows = training arm, columns = evaluation input.

| train ↓ / eval → | raw | masked | spread |
|---|---|---|---|
| baseline | 0.7588 | 0.7443 | **−0.0145** |
| masked | 0.7623 | **0.7667** | +0.0044 |
| maskaug | 0.7646 | 0.7641 | −0.0005 |

Teach the reader to read the table (one short paragraph), then make two points:

1. **`masked/raw` (0.7623) > `baseline/raw` (0.7588).** The masked-trained model beats the
   baseline on *unmasked* images — a condition it never trained on. This is evidence
   against the "masking merely aligns train and test distributions" confound, which is the
   whole reason the 2×2 exists. (Note honestly: the difference is +0.0035 and the CIs
   overlap heavily.)
2. **The baseline is condition-sensitive; the masked arms are not.** Read the spread column.
   Removing occluders at test time costs the baseline 1.45 GAP points; the masked arm gains
   slightly and maskaug is flat to within 0.05 points. This asymmetry is the largest effect
   anywhere in the results and sets up §V-D.

### C. The stated hypothesis is not supported

**Table 4.** Matched-condition GAP by occlusion stratum, with 95% bootstrap CIs
(1,000 resamples, paired).

| stratum | n | baseline/raw | masked/masked | Δ | Δ 95% CI | p |
|---|---|---|---|---|---|---|
| none | 6,521 | 0.7524 | 0.7589 | +0.0065 | [−0.0034, +0.0158] | 0.215 |
| low | 1,039 | 0.8075 | 0.8245 | +0.0170 | [−0.0027, +0.0370] | 0.098 |
| medium | 317 | 0.7689 | 0.7733 | +0.0044 | [−0.0411, +0.0461] | 0.848 |
| high | 123 | 0.6475 | 0.6587 | +0.0112 | [−0.0752, +0.0963] | 0.764 |
| **ALL** | 8,000 | 0.7588 | 0.7667 | +0.0079 | [−0.0010, +0.0163] | 0.084 |

And for `maskaug` (report the raw column — its matched condition is genuinely ambiguous
since it trained at p=0.5): ALL Δ +0.0058, CI [−0.0027, +0.0133], p=0.189.
McNemar, masked vs baseline: 548 fixed, 508 broke, p=0.230.

**Say it plainly.** Deltas are positive in every stratum, never significant, and show no
dose–response. Do **not** call p=0.084 "marginally significant." One sentence of
interpretation: the effect direction is consistent with the hypothesis, the magnitude is
not resolvable at this sample size with one seed per arm.

Add the structural observation: **81.4% of test images sit in the `none` stratum**, where
masking is a no-op by construction, so the overall figure is by design dominated by images
the intervention cannot affect.

### D. A large, monotone, significant asymmetric dependency

**This is the finding. Give it the most space in §V.**

**Table 5.** Change in GAP when a model is fed the input condition it was *not* trained on.

| stratum | n | baseline: raw→masked | 95% CI | p | masked: masked→raw | maskaug: raw→masked |
|---|---|---|---|---|---|---|
| none | 6,521 | +0.0006 | [−0.0006, +0.0017] | 0.369 | +0.0002 | −0.0000 |
| low | 1,039 | −0.0249 | [−0.0374, −0.0140] | <0.001 | −0.0057 | +0.0004 |
| medium | 317 | −0.1393 | [−0.1858, −0.0967] | <0.001 | −0.0485 | +0.0070 |
| high | 123 | **−0.4171** | [−0.5175, −0.3155] | <0.001 | −0.1286 | −0.0429 |

**Figure:** `gap_vs_stratum.png` — GAP vs stratum, one line per cell. The diverging fan is
the visual argument; make it the largest figure in the paper.

Four points, in order:

1. **Monotone and highly significant.** The baseline degrades progressively with occlusion
   when transient content is removed, reaching −0.4171 at `high` — four times the CI
   half-width, p<0.001. Absolute collapse: 0.6475 → **0.2304**, a 64% relative drop.
2. **The reverse mismatch is 3.5× smaller.** The masked model loses 0.1174 on raw input at
   `high` (p=0.007) — significant, but far less catastrophic.
3. **`maskaug` is nearly condition-invariant** (ALL: 0.7646 raw vs 0.7641 masked; worst
   stratum −0.0429, not significant). Exactly what stochastic masking should produce.
4. **Anticipate the obvious objection and refute it with your own data.** A reviewer will
   say: at `high` occlusion you blank 36.5% of the image on average, so *any* model would
   degrade — this is distribution shift, not shortcut reliance. Your counterfactual answers
   it: the masked model reaches **0.6587 without access to any transient pixels**,
   statistically indistinguishable from the baseline's 0.6475 *with* them
   (Δ +0.0112, CI [−0.0752, +0.0963]). **The information is not necessary.**

### E. Internal control — the `none` stratum

Short but essential. `baseline_masked` vs `baseline_raw` at `none`:
**Δ +0.0006, CI [−0.0006, +0.0017], p=0.369.**

Where there is nothing to mask, masking changes nothing. This rules out the most dangerous
alternative explanation available to a reviewer — that the `mean_fill` artefact itself
perturbs predictions independent of what it covers. **Without this control, Table 5 is not
credible.** State it as such.

### F. Qualitative analysis

**Figure:** `qualitative_examples.png` — a 2×4 grid built from `preds_*.parquet`.

- Top row: images the **baseline gets right and the masked arm gets wrong** — the cost of
  over-masking, direct evidence for the detector failures documented in §III-C.3.
- Bottom row: images the **masked arm fixes** — where removing occluders helped.

Show the *failure* row first. Showing what your method breaks is what separates an analysis
from a sales pitch, and McNemar says there are 508 such images against 548 fixes — a nearly
even trade that the reader deserves to see.

### G. Synthesis — state the conclusion in one sentence

> Masking is **sufficient** (transient content carries no task-necessary information),
> **safe** (no measurable accuracy cost), but not **beneficial** (no measurable accuracy
> gain). Its value is invariance, not accuracy.

Then the practical implication, which is the paper's payoff: a model deployed where occluder
statistics differ from training — different seasons, tourist demographics, vehicle fleets —
inherits the baseline's dependency as a **hidden failure mode**. Masking removes it at no
accuracy cost, and `maskaug` achieves nearly the same invariance at **+14.5%** per-epoch
cost instead of **+25.8%**, making it the practical recommendation.

## VI. Discussion & Limitations

Be specific; vague limitations sections read as filler. Seven items:

1. **Scale.** 1,000 classes / 80,000 images, balanced. Results may not transfer to 81,313
   classes with a natural long tail.
2. **Protocol.** Held-out train split, not the competition test set. No non-landmark
   distractors, whereas the real test set contains them — so absolute GAP is **not
   comparable** to published leaderboard numbers.
3. **Occlusion ratio is a proxy** derived from the very detector whose errors you
   documented. Circularity worth naming: images where the detector fails are also
   mis-stratified. A human-labelled subsample would be needed to validate the proxy.
4. **Categorical detector limitation.** COCO has no `statue`; the subject guard mitigates
   but does not solve it.
5. **Single seed per arm.** No estimate of run-to-run variance; differences smaller than
   seed noise cannot be resolved. State the required fix (3–5 seeds per arm) and why you
   could not (GPU budget).

6. **Statistical power — quantify it, do not just assert it.** This is your strongest
   limitation paragraph because it comes with a number. The `high` stratum CI half-width is
   **±0.095 at n=123**. Resolving the observed +0.011 effect needs the interval ~8.6×
   tighter, i.e. **≈9,100 high-occlusion test images**; at the measured 1.54% prevalence
   that is a test set of roughly **590,000 images**. So the study is not underpowered by
   accident — it is underpowered by the **rarity of the phenomenon**, and no Kaggle-scale
   experiment could resolve an effect this small. This reframes the null as a quantified
   boundary condition rather than a failure, and it directly motivates occlusion-enriched
   sampling in §VII.

7. **Occlusion is not randomly assigned — the strata differ in more than occlusion.**
   `low` outperforms `none` in *every* cell (0.8075 vs 0.7524 for baseline/raw). Lightly
   occluded images are **easier** than clean ones. The likely reason: photographs containing
   people are canonical, well-framed shots of famous landmarks, whereas `none` includes
   interiors, detail crops and oddities. Cross-stratum comparisons are therefore confounded;
   only *within*-stratum comparisons (which is what Tables 4 and 5 report) are clean. Say
   this explicitly — a careful reader will notice the inversion and you want to have named
   it first.

8. **The `maskaug` matched condition is ambiguous.** It trained at p=0.5, so neither raw nor
   masked is unambiguously "its" condition. Report both columns rather than choosing the
   flattering one.
9. **Fill-strategy confound.** Only `mean_fill` was run at full scale. `black`, `blur`,
   `inpaint_telea` remain unablated, so *"masking helps/doesn't help"* is really
   *"mean-fill masking helps/doesn't help."*
10. **Label noise at the extremes.** Images above ~0.5 occlusion are frequently group
   photographs with no visible landmark. Masking cannot fix a mislabelled image, and these
   images dilute the `high` stratum.

---

## VII. Future Work

Order by ratio of payoff to effort — that ordering is itself a contribution.

### Immediate, cheap, enabled by your stored detections

- **Fill-strategy ablation** (black / blur / inpaint vs mean-fill). Costs no GPU detection
  pass — this is *why* you stored detections rather than masked images.
- **Taxonomy ablation** — with/without the `animals` tier, with `street_furniture`, with
  `misc_objects`. Same argument.
- **Subject-guard ablation** — quantify the precision/recall trade you asserted.
- **Multi-seed runs** to bound seed variance and settle whether any observed difference is
  real.

### Methodological extensions

- **Feature-space rather than pixel-space masking.** Instead of overwriting pixels (which
  injects an artefact the network can learn), suppress the corresponding spatial positions
  in the backbone's feature map, or add the mask as a fourth input channel so the network is
  *told* which regions are unreliable rather than being handed a fabricated fill.
  **This is the most promising direction** — argue for it at some length.
- **Occlusion-aware loss weighting** — down-weight heavily occluded samples instead of
  masking them.
- **Curriculum** — train on clean images first, introduce occluded ones progressively.
- **Test-time masking ensemble** — average predictions over masked and raw views.

### Better occlusion estimation

- Fine-tune a segmentation model on landmark imagery with a `statue`/`monument` class,
  directly removing your worst failure mode.
- **Open-vocabulary detection** (text-prompted) to define transient classes by description
  rather than a fixed 80-class list.
- **Depth estimation** to separate *foreground* occluders from background structure
  geometrically, rather than by semantic class.

### Scaling and validation

- Full 81,313-class GLDv2; larger backbones and higher resolution; DOLG-style global–local
  descriptors.
- Add non-landmark distractors to match the real competition protocol.
- Human-annotated occlusion labels on a 1,000-image subsample to validate the automated
  ratio.

---

## Practical notes

**Abstract — 150–250 words, written last.**
Problem → what you built → scale (80k images, 1,000 classes, 3 arms) → the two headline
numbers (67.0% zero occlusion; your ΔGAP with CI) → the one-sentence finding.

**Figures and tables:** see the dedicated section below.

**Tables:** subset composition · taxonomy + thresholds · 2×2 GAP · stratified GAP with CIs ·
significance.

**Reproducibility statement** — one short paragraph: code on GitHub, frozen manifests
committed (`subset_splits.csv`, `class_map.csv`, `occlusion_index.csv`), single seed 42,
exact package versions. Cheap to write, and reviewers weight it.

---

## Figure & table placement

Six figures and five tables is right for 6–8 IEEE pages. **Place each figure in the section
that argues from it**, not in an appendix — a reader should never have to hunt.

`\begin{figure}` = single column. `\begin{figure*}[t]` = full width, floats to page top;
use it for the two wide ones marked below.

### Figures

| # | File | Section | Width | Why it goes there |
|---|---|---|---|---|
| 1 | `class_frequency.png` | III-B | single | Justifies the 80-image cap: even rank-1000 has ~200 images available |
| 2 | `taxonomy_ablation.png` | **III-C.3** | **`figure*`** | Your strongest figure. Needs full width — 3 columns × 6 rows |
| 3 | `occlusion_distribution.png` | III-C.4 | **`figure*`** | Two panels; already 9.2in wide, don't squeeze it |
| 4 | `masking_midrange.png` | III-D | single | Shows masking working in the band the hypothesis is about |
| 5 | `training_curves.png` | V-A | single | Three arms converge comparably — the negative control |
| 6 | `gap_vs_stratum.png` | **V-D** | **`figure*`** | The diverging fan. Make this the largest figure in the paper |

> `masking_examples.png` is the *pre-refinement* version of Figure 2. If you have room,
> use it as Fig. 2a ("permissive") with `taxonomy_ablation.png` as 2b — otherwise drop it,
> since the ablation figure already contains a permissive column.

### Captions

Write them so each figure is self-contained — a reader skimming figures should get the paper.

**Fig. 1**
> Image counts for the 1,000 selected landmark classes in the full GLDv2 training set,
> ranked by frequency (log scale). All selected classes exceed the 80-image sampling cap,
> yielding a perfectly balanced 80,000-image subset.

**Fig. 2** *(figure\*)*
> Effect of taxonomy refinement on masking behaviour. Left: original image. Centre:
> permissive configuration (all COCO tiers, uniform 0.25 confidence threshold, no subject
> guard). Right: final configuration (narrowed taxonomy, per-class thresholds, dominant-
> subject guard). The permissive configuration masks the landmark itself in three of the six
> cases — a building facade detected as *boat*, a stone statue as *person*, and a fireworks
> display as *bird/car/person*. Occlusion ratios shown per panel.

**Fig. 3** *(figure\*)*
> Distribution of transient occlusion across the 80,000-image subset, measured as the
> fraction of image area covered by the union of YOLOv8m-seg detections in the transient
> taxonomy. (a) Counts per evaluation stratum. (b) Distribution restricted to the 26,425
> images with non-zero occlusion, log count axis, dashed lines marking stratum boundaries.
> 67.0% of images contain no detected transient content; the `high` stratum (ratio ≥ 0.25)
> holds 1,062 images, of which 123 fall in the test split.

**Fig. 4**
> Masking applied to mid-range occlusion images (ratio 0.08–0.25), the band in which the
> landmark remains visible after transient content is removed. Top: raw input. Bottom:
> `mean_fill` masked input, as seen by the model after augmentation.

**Fig. 5**
> Validation GAP per epoch for the three training arms. Training loss trajectories are
> near-identical across arms (final 0.28 / 0.29 / 0.30), indicating masking neither
> increased task difficulty nor removed learnable signal. Note each arm validates on its own
> input condition, so the curves are not directly comparable; see Table 3.

**Fig. 6** *(figure\*)*
> Test-set GAP by occlusion stratum for each (training arm, evaluation input) pair, with 95%
> bootstrap confidence intervals. The baseline degrades monotonically as occlusion increases
> when transient regions are removed at test time (−0.4171 at the `high` stratum, p<0.001),
> while the masked and stochastically-masked arms remain substantially more stable. Stratum
> sizes: none 6,521, low 1,039, medium 317, high 123.

### Tables

| # | Content | Section | Width | Source |
|---|---|---|---|---|
| 1 | Subset composition & splits | III-B | single | §III-B of this guide |
| 2 | Transient taxonomy + per-class thresholds | III-C.2 | single | §III-C.2 |
| 3 | 2×2 cross-condition GAP | V-B | single | `cross_condition_gap.csv` |
| 4 | Matched-condition GAP by stratum, with CIs | V-C | **`table*`** | `stratified_gap_ci.csv` + `stratified_delta_ci.csv` |
| 5 | Mismatch penalty by stratum, with CIs | V-D | **`table*`** | `stratified_delta_ci.csv` |

Tables 4 and 5 carry n, GAP, CI and p per row, so they need full width. Use `booktabs`
(`\toprule` / `\midrule` / `\bottomrule`), no vertical rules — that is the IEEE house style
and it reads far better than the default.

### Two figures you still have to generate

Neither needs a GPU.

**`training_curves.png`** — from the three `history.json` files in
`artifacts/runs/<arm>/`, each holding per-epoch `train_loss`, `val_gap`, `val_top1`.
Plot val GAP vs epoch, one line per arm, and mark each arm's selected epoch.

**`gap_vs_stratum.png`** — from `stratified_gap_ci.csv`. X = stratum (none/low/medium/high),
Y = GAP, one line per cell with CI bands. Six lines is too many to read at once: plot the
three **matched** cells as solid lines and the three **mismatched** cells as dashed, or
facet into two panels (matched | mismatched). The diverging fan at the right-hand end is the
entire visual argument — do not let it get lost in a legend.

---

## Quick-reference: every number in one place

```
DATA
  GLDv2 train              1,580,470 images / 81,313 classes / ~98 GB
  Subset                   1,000 classes, 80 images each = 80,000
  Splits                   64,000 train / 8,000 val / 8,000 test (64/8/8 per class, sd 0.0)

DETECTION
  Model                    YOLOv8m-seg, imgsz 640, conf 0.10, iou 0.70, max_det 100
  Taxonomy                 20 COCO classes / 4 tiers
  Never masked             airplane, train, clock, boat
  Thresholds               person 0.30 | umbrella+vehicles 0.40 | default 0.55
  Subject guard            area >= 0.25 AND centre within 0.30 of image centre
  Dilation                 4 px
  Median detection area    0.002 of frame
  Diagnostic (200 imgs)    48 boat, 7 giraffe, 1 elephant false positives

CACHE
  Images                   256px short side, JPEG q90 -> 2.40 GB
  Masks                    binary PNG, written only when non-empty -> 32.8 MB
  Non-empty masks          26,425 (33.0%)
  Detections               224 MB parquet
  Build time               ~75 min, one T4 pass

OCCLUSION
  mean 0.0189 | median 0.000 | max 0.816 | zero 67.0%
  none   < 0.02       65,115 (81.4%)   test 6,521
  low    0.02-0.10    10,436 (13.0%)   test 1,039
  medium 0.10-0.25     3,387 ( 4.2%)   test   317
  high   >= 0.25       1,062 ( 1.3%)   test   123

MODEL
  efficientnet_b0.ra4_e3600_r224_in1k, 5.18 M params
  512-d embedding, ArcFace scale 30 margin 0.30, dropout 0.2
  224px, batch 64, AdamW lr 3e-4 wd 1e-4, cosine + 1 epoch warmup
  15 epochs, AMP, grad clip 5.0, select on val GAP, patience 5
  ~5.1 min/epoch on T4

ARMS
  baseline  raw
  masked    mean_fill, p = 1.0
  maskaug   mean_fill, p = 0.5
  max_mask_fraction 0.85

EVALUATION
  GAP (global ranking), 2x2 cross-condition, stratified by occlusion bin
  Bootstrap 1,000 iters; paired bootstrap on delta; exact-binomial McNemar
  Test split only; checkpoints selected on val
  Seed 42 throughout

VAL (each arm on its OWN condition -- not comparable across arms)
  baseline  0.7662 @ epoch 14   310 s/epoch
  masked    0.7507 @ epoch 12   390 s/epoch   (+25.8% cost)
  maskaug   0.7603 @ epoch 14   355 s/epoch   (+14.5% cost)

TEST 2x2 GAP (8,000 images)
                  eval:raw   eval:masked   spread
  baseline          0.7588      0.7443     -0.0145
  masked            0.7623      0.7667     +0.0044
  maskaug           0.7646      0.7641     -0.0005

HEADLINE COMPARISONS (paired bootstrap, 1,000 iters)
  masked  vs baseline  dGAP +0.0079  CI [-0.0010, +0.0163]  p=0.084  NOT sig
  maskaug vs baseline  dGAP +0.0053  CI [-0.0031, +0.0133]  p=0.236  NOT sig
  McNemar masked:  548 fixed / 508 broke, p=0.230
  McNemar maskaug: 519 fixed / 488 broke, p=0.344

STRATIFIED GAP, matched conditions (baseline/raw vs masked/masked)
  stratum    n     base    masked   delta    CI                  p
  none     6521   0.7524   0.7589  +0.0065  [-0.0034,+0.0158]  0.215
  low      1039   0.8075   0.8245  +0.0170  [-0.0027,+0.0370]  0.098
  medium    317   0.7689   0.7733  +0.0044  [-0.0411,+0.0461]  0.848
  high      123   0.6475   0.6587  +0.0112  [-0.0752,+0.0963]  0.764
  ALL      8000   0.7588   0.7667  +0.0079  [-0.0010,+0.0163]  0.084

MISMATCH PENALTY  <-- THE FINDING (baseline raw->masked)
  none     +0.0006  CI [-0.0006,+0.0017]  p=0.369   (internal control: no effect)
  low      -0.0249  CI [-0.0374,-0.0140]  p<0.001
  medium   -0.1393  CI [-0.1858,-0.0967]  p<0.001
  high     -0.4171  CI [-0.5175,-0.3155]  p<0.001   0.6475 -> 0.2304
  reverse (masked masked->raw) at high: -0.1174, p=0.007  (3.5x smaller)
  maskaug raw->masked at high: -0.0429, not significant  (most invariant)

POWER
  high-stratum CI half-width +-0.095 at n=123
  to resolve +0.011 needs ~9,100 high images -> ~590,000 test images at 1.54% prevalence

CONCLUSION IN ONE LINE
  Masking is SUFFICIENT (no necessary info in transient regions),
  SAFE (no accuracy cost), NOT BENEFICIAL (no accuracy gain).
  Its value is invariance, not accuracy. maskaug = best cost/robustness trade.
```
