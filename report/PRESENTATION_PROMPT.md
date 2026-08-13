# Prompt for Gemini Pro — 10–12 minute presentation

> Copy everything below the line into Gemini Pro, and attach the images listed in the
> "Image manifest" section in the same message.

---

You are helping me build a **10–12 minute conference-style presentation** for a graduate
computer vision course final project. I will attach my figures as images; use them where the
manifest below says.

## Who I am and what this is

I am Siddharth Mehta, a graduate student at Northeastern University. The project extends the
Kaggle Google Landmark Recognition 2021 challenge to study whether occlusion-aware
preprocessing improves landmark recognition. The audience is my instructor and classmates:
technically literate in deep learning, but not specialists in landmark retrieval. They will
be assessing experimental rigour, not novelty of architecture.

## What I need you to produce

**15 slides totalling 10–12 minutes.** For each slide give me:

1. **Slide title** — a claim, not a label. "Masking does not improve accuracy" beats "Results".
2. **Slide body** — at most 5 bullets, at most 12 words each. Presentations are read, not
   studied. Put numbers on the slide only when the number *is* the point.
3. **Speaker notes** — 80–130 words of what I actually say, in spoken register (contractions
   fine, no bullet fragments). This is the part I will rehearse from.
4. **Timing** — target seconds for the slide. The totals must sum to 630–720 seconds.
5. **Visual** — which attached image to use, or a description of a simple diagram/table to
   build if no image applies.

Then, separately:

6. **A Q&A preparation section** — 6 likely questions with 2–3 sentence answers, including
   the hostile ones (see "Anticipated objections" below).

## The narrative arc — this is the most important instruction

The honest story is: **the original hypothesis failed, and a more interesting finding
replaced it.** Structure the talk that way and do not hide the null result.

- Slides 1–4: motivate the problem and the approach.
- Slides 5–9: methodology, with a **substantial detour into a detector failure I discovered
  and fixed** — this is the most engaging part of the talk and should not be compressed.
- Slide 11: **state plainly that masking did not significantly improve accuracy.** Do not
  soften it, do not call p = 0.084 "marginally significant", do not lead with a
  cherry-picked stratum.
- Slides 12–13: pivot to the finding that *did* survive statistical testing — a large,
  monotone, highly significant asymmetry — and the control that proves it is real.
- Slides 14–15: limitations and conclusion.

**Critical framing constraint:** never describe the project as successful at its original
goal. Describe it as a study that produced a clear, well-supported answer, which happened to
be "no" to the first question and "yes, and here's something better" to a second one. A
presentation that oversells a null result reads as dishonest to an experienced audience; one
that owns it reads as competent.

---

# PROJECT DATA — use these numbers exactly, do not invent or round differently

## Problem framing

Google Landmarks Dataset v2 is crowdsourced from Wikimedia, so images are tourist
photographs and contain tourists, vehicles, animals and personal effects. In instance-level
recognition the class is a *specific physical structure*, so anything not attached to that
structure is not evidence for the label. Risk: a model can learn features that co-occur with
a landmark in the dataset but not in the world — shortcut learning. Existing competition
solutions address this model-side (ArcFace, DOLG, re-ranking); none removes the content.
Published results also report only aggregate GAP, so nobody reports whether gains are uniform
across occlusion levels.

## Dataset and subset design

- Source: GLDv2 / Landmark Recognition 2021 — 1,580,470 images, 81,313 classes, ~98 GB.
- Competition test labels were never released and the challenge is closed (the public test
  folder holds ~10 placeholder images), so **all evaluation uses a held-out split of train**.
- Subset: top 1,000 classes by frequency, capped at 80 images each → **exactly 80,000 images**.
- Split per class: 64 / 8 / 8 → **64,000 train / 8,000 val / 8,000 test**, standard deviation 0.
- Rank-1 class has 6,272 images available; rank-1000 has 142 — both above the 80 cap, so the
  subset is perfectly balanced and class imbalance is eliminated as a confound.
- Frequency ranking is deliberate: the most photographed landmarks are the most occluded, so
  this *enriches* the phenomenon rather than diluting it.

## Detection pipeline

- YOLOv8m-seg, 640 px, confidence 0.10, IoU 0.70, max 100 detections per image.
- **Segmentation not bounding boxes**: a box around a person in front of a cathedral deletes
  a rectangle of cathedral too.
- **Detect permissively, filter offline**: store every detection down to conf 0.10 as
  normalised polygons; apply taxonomy, thresholds and guard at mask-render time. Ablations
  then cost CPU seconds instead of another GPU pass over 80,000 images.
- One fused GPU pass: ~75 minutes on an NVIDIA T4, producing a 2.40 GB image cache, 32.8 MB
  of binary masks, and a 224 MB detection record.

## Transient taxonomy — 20 COCO classes in 4 tiers

| Tier | Classes |
|---|---|
| people | person |
| vehicles | bicycle, car, motorcycle, bus, truck |
| animals | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| portable objects | backpack, umbrella, handbag, suitcase |
| **never masked** | airplane, train, clock, boat |

The never-mask list exists because each of those classes *can be the landmark* — museum
aircraft, preserved locomotives, Big Ben, historic ships.

## The detector failure (give this a full slide — it's the best story in the talk)

An initial configuration (YOLOv8s-seg, uniform 0.25 threshold, no geometric filter) produced
damaging false positives. On a 200-image diagnostic sample: **48 `boat` detections, 7
`giraffe`, 1 `elephant`** in landmark photographs.

| Image content | Detected as | Consequence |
|---|---|---|
| Building facade | boat, kite | Landmark masked out entirely |
| Stone statue | person | Landmark masked out |
| Fireworks display | bird, car, person | Subject masked out |
| Rock formation | spurious | Subject partly masked |

Root cause: **COCO has no `statue` class**, and carved human figures fall inside the learned
`person` distribution. This is a categorical limitation of transfer, not a tunable threshold.

## Three mitigations

1. **Per-class confidence thresholds.** Measured medians were 0.16–0.33 with maxima near
   0.95, so false positives lived in the low-confidence tail. `person` 0.30, vehicles and
   `umbrella` 0.40, everything else 0.55.
2. **Taxonomy narrowing.** COCO sports/tableware classes (frisbee, kite, sports ball, bottle,
   wine glass) fired on architecture and were moved to an opt-in tier. `boat` moved to
   never-mask.
3. **Dominant-subject guard.** Reject any detection covering ≥25% of the frame centred within
   0.30 of image centre. Principle: *an occluder is not the photographic subject.* Justified
   empirically — median detection area is **0.002** of the frame, two orders of magnitude
   below the threshold, so the guard fires only on the extreme tail.

Effect: fireworks 0.36 → 0.00, rock formation 0.13 → 0.00, statue 0.39 → 0.10. Residual false
positives remain on the Lotus Temple dome and Sydney Opera House roof.

## Occlusion ratio and strata

Occlusion ratio = union area of retained transient regions (dilated 4 px) ÷ image area.
Mean 0.0189, median 0.000, max 0.816, and **67.0% of images have exactly zero**.

| Stratum | Range | All 80k | % | Test |
|---|---|---|---|---|
| none | < 0.02 | 65,115 | 81.4 | 6,521 |
| low | 0.02 – 0.10 | 10,436 | 13.0 | 1,039 |
| medium | 0.10 – 0.25 | 3,387 | 4.2 | 317 |
| high | ≥ 0.25 | 1,062 | 1.3 | 123 |

## Masking strategy

Four fills implemented (black, mean fill, blur, Telea inpainting); **mean fill** used
throughout. Black is a poor default: a solid black region is itself a learnable signal, so a
network can key on mask *shape* rather than ignoring the region. Mean fill uses the mean of
the **retained** pixels only. Masking skipped above 85% coverage. Masking applied before
augmentation.

## Experimental arms

| Arm | Training input | Role |
|---|---|---|
| baseline | raw | control |
| masked | transient regions removed, p = 1.0 | treatment |
| maskaug | masked with p = 0.5 | treatment as stochastic augmentation |

Identical code, config, seed 42, splits, schedule and augmentation. Only the input transform
differs. Masking is a runtime dataset argument, which is what makes the 2×2 evaluation
possible.

## Model and training

EfficientNet-B0 (`efficientnet_b0.ra4_e3600_r224_in1k`), 5.18 M parameters, 512-d embedding,
ArcFace head (scale 30, margin 0.30). 224 × 224 inputs, AdamW lr 3e-4, weight decay 1e-4,
cosine schedule with 1 epoch warmup, batch 64, mixed precision, 15 epochs, checkpoint
selected on validation GAP. NVIDIA T4.

**Metric — GAP (Global Average Precision):** predictions across *all* images are ranked
globally by confidence, so a confidently wrong prediction pollutes precision for every
lower-ranked prediction. Confidence calibration is therefore part of the metric. Verified on
a synthetic example: one confident error scores 0.389 versus 0.667 for the same error made
uncertainly.

**Statistics:** paired bootstrap, 1,000 resamples, identical indices for both arms; plus
McNemar's test on per-image top-1 correctness.

## Training results

| Arm | Best val GAP | Epoch | Val condition | s/epoch | overhead |
|---|---|---|---|---|---|
| baseline | 0.7662 | 14 | raw | 310 | — |
| masked | 0.7507 | 12 | masked | 390 | +25.8% |
| maskaug | 0.7603 | 14 | masked | 355 | +14.5% |

Final training losses nearly identical (0.28 / 0.29 / 0.30 from ~13.5) — a negative control
showing masking neither made the task harder to fit nor removed learnable signal.
**These three GAP values are NOT comparable** — each arm validated on its own input
condition.

## RESULT 1 — Cross-condition matrix (test set, 8,000 images)

| train ↓ / eval → | raw | masked | spread |
|---|---|---|---|
| baseline | 0.7588 | 0.7443 | −0.0145 |
| masked | 0.7623 | **0.7667** | +0.0044 |
| maskaug | 0.7646 | 0.7641 | −0.0005 |

Key reading: `masked/raw` (0.7623) exceeds `baseline/raw` (0.7588) — the masked model does
better on an input condition it never trained on, which argues against pure train/test
distribution alignment. But the margin is small and CIs overlap.

## RESULT 2 — The stated hypothesis is NOT supported

Matched-condition GAP by stratum, paired bootstrap CIs:

| stratum | n | baseline/raw | masked/masked | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| none | 6,521 | 0.7524 | 0.7589 | +0.0065 | [−0.0034, +0.0158] | 0.215 |
| low | 1,039 | 0.8075 | 0.8245 | +0.0170 | [−0.0027, +0.0370] | 0.098 |
| medium | 317 | 0.7689 | 0.7733 | +0.0044 | [−0.0411, +0.0461] | 0.848 |
| high | 123 | 0.6475 | 0.6587 | +0.0112 | [−0.0752, +0.0963] | 0.764 |
| **ALL** | 8,000 | 0.7588 | 0.7667 | **+0.0079** | **[−0.0010, +0.0163]** | **0.084** |

Positive in every stratum, significant in none, no dose–response. McNemar: masked arm fixed
548 baseline errors and introduced 508 (p = 0.230). maskaug on raw input: +0.0058,
CI [−0.0027, +0.0133], p = 0.189.

Structural reason an overall effect was always unlikely: **81.4% of test images are in the
`none` stratum**, where masking is a no-op by construction.

## RESULT 3 — The finding that DID survive (make this the centrepiece)

Change in GAP when a model gets the input condition it was **not** trained on:

| stratum | n | baseline: raw→masked | 95% CI | p | masked: masked→raw | maskaug: raw→masked |
|---|---|---|---|---|---|---|
| none | 6,521 | +0.0006 | [−0.0006, +0.0017] | 0.369 | +0.0002 | −0.0000 |
| low | 1,039 | −0.0249 | [−0.0374, −0.0140] | <0.001 | −0.0057 | +0.0004 |
| medium | 317 | −0.1393 | [−0.1858, −0.0967] | <0.001 | −0.0485 | +0.0070 |
| high | 123 | **−0.4171** | **[−0.5175, −0.3155]** | **<0.001** | −0.1286 | −0.0429 |

- Baseline collapses from **0.6475 → 0.2304** at high occlusion — a 64% relative drop, about
  four times the CI half-width.
- Reverse mismatch is 3.5× smaller (masked model: −0.1174 on raw at high, p = 0.007).
- maskaug is nearly condition-invariant (worst stratum −0.0429, not significant).

**The counterfactual that defeats the obvious objection:** a critic will say this is just
distribution shift, since masking blanks 36.5% of the image at high occlusion, and any model
would degrade. Answer: the masked model reaches **0.6587 without access to any transient
pixels**, statistically indistinguishable from the baseline's 0.6475 *with* them
(Δ +0.0112, CI [−0.0752, +0.0963]). If those pixels carried necessary information, no model
trained without them could match one trained with them.

## RESULT 4 — Internal control

`baseline/masked` vs `baseline/raw` on the 6,521 `none`-stratum images:
**Δ +0.0006, CI [−0.0006, +0.0017], p = 0.369.**

Where there is nothing to mask, masking changes nothing. This rules out the alternative
explanation that the mean-fill artefact itself perturbs predictions. Without this control,
Result 3 would not be attributable to removing transient content.

## Conclusion — the three-part claim

> Masking is **sufficient** (transient content carries no task-necessary information),
> **safe** (no measurable accuracy cost), but **not beneficial** (no measurable accuracy
> gain). Its value is invariance, not accuracy.

Practical implication: a model trained on raw crowdsourced data acquires a dependency on
occluder content that is invisible under standard evaluation, because evaluation data shares
the training data's occluder statistics. Deployed where those statistics differ — different
season, visitor demographic, vehicle fleet — that dependency becomes a hidden failure mode.
maskaug achieves nearly the same invariance at +14.5% per-epoch cost instead of +25.8%, and
is the practical recommendation.

## Limitations (compress to one slide — pick the 4 strongest)

- **Statistical power, quantified.** High-stratum CI half-width is ±0.095 at n = 123.
  Resolving the observed +0.011 effect needs ~9,100 high-occlusion test images — at 1.54%
  prevalence, a test set of ~593,000 images. The study is underpowered by the **rarity of the
  phenomenon**, not by design error.
- **Occlusion is not randomly assigned.** The `low` stratum outperforms `none` in every cell
  (0.8075 vs 0.7524). Lightly occluded images are *easier*, probably because photos with
  people are canonical well-framed shots. Cross-stratum comparison is confounded; only
  within-stratum comparison is clean.
- **Single seed per arm** — no estimate of run-to-run variance.
- **Occlusion ratio is a proxy** derived from the same detector whose failures we documented
  (circularity).
- Also available if needed: scale (1,000 of 81,313 classes), held-out train split rather than
  competition test set, only mean-fill ablated, label noise above 0.5 occlusion.

## Future work (one slide, 3–4 items max)

- **Feature-space masking** instead of pixel-space — suppress positions in the backbone
  feature map, or pass the mask as a fourth input channel, so the network is *told* which
  regions are unreliable rather than handed a fabricated fill. This removes the artefact the
  network can otherwise learn. **Lead with this one.**
- Fine-tune segmentation on landmark imagery with an explicit `statue`/`monument` class —
  directly fixes the worst failure mode.
- Occlusion-enriched test sampling with reweighting to natural prevalence — resolves the same
  effect with a fraction of the compute, per the power analysis.
- Cheap ablations already enabled by the stored detections: fill strategy, taxonomy tiers,
  subject guard, multi-seed.

---

# Image manifest

I am attaching these images. Use each on the slide indicated.

| Image | Use on slide about | Notes for the slide |
|---|---|---|
| `class_frequency.png` | Subset design | Log-scale class frequency, top 1,000. Shows all classes clear the 80-image cap |
| `taxonomy_ablation.png` | **The detector failure + fix** | 3 columns (raw / permissive / strict) × 6 rows. **Give this a full slide** — it is the most compelling visual in the deck |
| `occlusion_distribution.png` | Occlusion measurement | Two panels: stratum counts, and the non-zero tail on a log axis with stratum cuts marked |
| `masking_examples.png` | *Optional* — the original failures | Use only if there is room; the ablation figure already contains a permissive column |
| `masking_midrange.png` | What masking actually does | Raw vs masked for occlusion 0.08–0.25, the band where the hypothesis lives |
| `training_curves.png` | Training dynamics | Val GAP per epoch, three arms |
| `gap_vs_stratum.png` | **Result 3** | GAP by stratum per cell with CI bands. The diverging fan is the visual argument — **make it the largest visual in the deck** |

If I have not attached one of these, design the slide around a table or a simple diagram
instead, and tell me which image is missing.

---

# Anticipated objections — cover these in the Q&A section

1. *"Your result isn't significant, so what did you actually show?"*
2. *"Isn't the baseline collapse just distribution shift? You blanked 36% of the image."*
3. *"Why didn't you use the real competition test set?"*
4. *"One seed per arm — how do you know this isn't noise?"*
5. *"Why is the `low` stratum easier than the `none` stratum? That seems backwards."*
6. *"If masking doesn't improve accuracy, why would anyone use it?"*

---

# Style constraints

- **No filler slides.** No "Outline", no "Thank you", no "Any questions?" slide.
- **Every slide title is a claim.** If a title could head any project's slide, rewrite it.
- **Numbers on slides only where the number is the argument.** Full tables go in speaker
  notes or backup slides, not on the main deck.
- **Speaker notes in spoken register.** I am reading them aloud, not silently.
- **Do not use the words "revolutionary", "novel framework", or "significantly improves"**
  anywhere. The last one is factually wrong for this project.
- Include a **backup slides** section (3–4 slides) with the full stratified tables and the
  power analysis, for questions.
