# Autoresearch: Bird Species Classifier

## What this is

A ResNet-50 fine-grained bird species classifier trained on the NABirds dataset (98 target species, ~10.5k train / ~1.2k val images). Images are cropped to bounding boxes, resized/padded to 240x240, and normalised with ImageNet statistics.

The training uses a progressive unfreezing strategy:
1. **Stage 1** -- train the FC head only (backbone frozen)
2. **Stage 2** -- unfreeze `layer4` + FC
3. **Stage 3** -- unfreeze `layer3` + `layer4` + FC
4. **Stage 4** *(optional)* -- unfreeze `layer2` + above
5. **Stage 5** *(optional)* -- unfreeze `layer1` + above (near full fine-tune)

Stages are fully configurable — you can use any combination of `"layer1."`, `"layer2."`, `"layer3."`, `"layer4."`, `"fc."` as unfreeze prefixes, in any order.

Each run is constrained to a **fixed wall-clock time budget** (default 30 minutes) so that all experiments are directly comparable regardless of architectural changes. Training stops at the budget or when all stages complete, whichever comes first.

## Optimization target

**Maximize `top1_val_acc`** within the fixed time budget.

This is the single number reported in `artifacts/autoresearch_log.csv`. The goal is to push this as high as possible through systematic experimentation.

## Files

| File | Role | Modifiable? |
|------|------|-------------|
| `train.py` | Training script -- the agent iterates on this | **YES** |
| `prepare.py` | One-time dataset setup | NO |
| `program.md` | These instructions | NO |
| `artifacts/autoresearch_log.csv` | Experiment log (append-only) | Read only |
| `artifacts/autoresearch_best.pt` | Best checkpoint (auto-saved) | Managed by train.py |
| `artifacts/autoresearch_splits.pkl` | Cached data splits | Managed by prepare.py |

## What you CAN do

Modify `train.py` -- this is the **only file you edit**. Everything in the CONFIGURATION section is fair game:

**Basic options:**
- **Augmentation pipeline**: `CROP_SCALE_MIN/MAX`, `JITTER_*`, `RANDOM_ERASING_*`
- **Learning rates**: per-stage `lr` values in the `STAGES` list
- **Stage structure**: number of stages, which layers to unfreeze (`"layer1."` through `"layer4."` + `"fc."`), max epochs per stage. You can add deeper stages unfreezing `layer2` and `layer1` for more aggressive fine-tuning at a lower LR (e.g. `1e-5`).
- **Dropout rate**: `DROPOUT`
- **Backbone**: `BACKBONE` -- swap to `"efficientnet_b0"` or `"mobilenet_v3_large"` (support is already wired in)
- **Label smoothing**: `LABEL_SMOOTHING`
- **Optimizer**: `OPTIMIZER` -- `"adam"`, `"adamw"`, or `"sgd"` (with `MOMENTUM`)
- **Scheduler**: `SCHEDULER` -- `"none"`, `"cosine"`, or `"step"`
- **Batch size**: `BATCH_SIZE`
- **Mixed precision**: `USE_AMP`
- **Time budget**: `TIME_BUDGET_SEC` (only if explicitly instructed)
- **NOTES**: always update this to describe what you changed

**Pre-staged advanced techniques (all OFF by default — toggle to enable):**
- **CutMix**: `CUTMIX_ALPHA` -- set >0 (try 1.0) to paste random patches between images. Forces the model to classify from partial views, preventing over-reliance on a single region. Critical for fine-grained.
- **Mixup**: `MIXUP_ALPHA` -- set >0 (try 0.2) to blend pairs of images and labels. When both CutMix and Mixup are >0, each batch randomly gets one or the other.
- **TrivialAugmentWide**: `USE_TRIVIAL_AUGMENT = True` -- replaces ColorJitter with a parameter-free augmentation that randomly selects one operation per image. Zero tuning needed.
- **GeM pooling**: `USE_GEM_POOLING = True` -- replaces average pooling with learnable Generalized Mean pooling. Focuses on strongly activated (discriminative) regions. +1-2% for fine-grained.
- **EMA**: `USE_EMA = True`, `EMA_DECAY` -- maintains smoothed model weights. The EMA model is used for evaluation and checkpointing. Reliable regularizer for small datasets.
- **Gradient clipping**: `GRAD_CLIP_NORM` -- set >0 (try 1.0) to clip gradient norms. Stabilizes training when unfreezing pretrained layers.
- **TTA**: `USE_TTA = True` -- horizontal flip test-time augmentation at final evaluation. Free +0.5-1% accuracy (only costs 2x inference, no training cost).
- **Warmup**: `WARMUP_EPOCHS` -- set >0 (try 2) for linear LR warmup at the start of each stage. Prevents large destructive updates when unfreezing new layers.
- **LLRD**: `LLRD_DECAY` -- set >0 (try 0.8) for layer-wise learning rate decay. Earlier ResNet layers get exponentially lower LR, preserving transferable features. ResNet-50 only.
- **Focal loss**: `USE_FOCAL_LOSS = True`, `FOCAL_GAMMA` -- down-weights easy examples, focuses on confusing species pairs.

You may also modify `build_transforms()`, `build_model()`, `build_optimizer()`, `build_scheduler()`, and the training loop functions for more structural changes.

## What you CANNOT do

- Modify `prepare.py` -- dataset loading logic is fixed.
- Modify the `# === EVALUATION & LOGGING` block at the bottom of `train.py` -- this ensures consistent logging and the structured summary output.
- Modify the constants section (`SEED`, `TARGET_SIZE`, `IMAGENET_*`, paths).
- Modify `artifacts/autoresearch_log.csv` -- only append via train.py.
- Install new packages or add dependencies.

## The goal is simple: get the highest `top1_val_acc`.

Since the time budget is fixed, you don't need to worry about training time -- it's always the same budget. Everything in the CONFIGURATION section is fair game: change the augmentation, the optimizer, the learning rates, the stage structure, the batch size, the backbone. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

## Output format

Once the script finishes it prints a structured summary:

```
---
top1_val_acc:     0.906518
top1_test_acc:    0.898148
training_seconds: 1187.3
total_epochs:     24
peak_memory_mb:   1234.5
time_budget_sec:  1800
status:           keep
notes:            cap stage1 at 6 epochs to reach stage2
analysis:         val_still_improving; improved(delta=+0.0123); train_acc=0.8912,val_acc=0.9065,best_at_epoch_23/24
```

The `analysis` field is auto-generated and flags issues like:
- `OVERFITTING(gap=0.12)` — train/val accuracy gap too large, needs more regularization
- `val_peaked_epoch_5(decline=0.015)` — val peaked early then fell, sign of overfitting after that point
- `val_still_improving` — accuracy was still going up when budget ran out, consider more epochs
- `plateau(range=0.003)` — val barely changed over last 4 epochs, diminishing returns
- `UNDERFITTING` — both accuracies low, model capacity or LR may be too limited
- `REGRESSED(delta=-0.02)` — this change hurt, revert it
- `hit_time_budget` — training was cut short

**Use the analysis to guide your next experiment.** For example: if you see OVERFITTING, try more dropout, label smoothing, or CutMix. If you see val_still_improving, try increasing max_epochs or reallocating time budget across stages.

You can extract the key metric:

```
grep "^top1_val_acc:" run.log
```

## The experiment loop

LOOP FOREVER:

1. **Read** `artifacts/autoresearch_log.csv` to understand the current best accuracy, what has been tried, and what the **analysis** column says about each run.
2. **Propose** one experiment. Use the `analysis` column to inform — but not constrain — your choice. The analysis flags symptoms (overfitting, plateau, time pressure); your job is to decide the best intervention, which may be a direct fix for the symptom OR a creative leap to something untried. Don't just iterate on the last failure — look at the full history for patterns and unexplored regions of the search space.
3. **Edit** `train.py` -- make the change and update `NOTES` to describe it.
4. **Run** `python train.py` and wait for it to complete.
5. **Read** the output to see the result (check the `---` summary block or the last line of `artifacts/autoresearch_log.csv`). **Pay attention to the `analysis` field.**
6. **Decide**: if `top1_val_acc` improved, keep the change. If it regressed, revert `train.py` to the previous version.
7. **Repeat** from step 1.

### Guidelines

- **One hypothesis at a time.** You may change 2-3 tightly coupled parameters when needed to test a single training-dynamics idea, but do not bundle unrelated changes.
- **Allowed coupled changes.** Examples: shorten a stage and raise that stage's LR to compensate; increase batch size and adjust LR accordingly; reallocate time across stages and retune only the affected stage LRs.
- **Disallowed bundles.** Do not change optimizer, scheduler, dropout, augmentation, and stage structure all in one run.
- **Keep NOTES concise and descriptive.** Good: `"adamw lr=3e-4, cosine schedule"`. Bad: `"trying something new"`.
- **If you make a coupled change, say so in NOTES.** Example: `"shorter stage1 + higher stage1 lr"`.
- **Read the log before every run.** The history tells you what has and hasn't worked.
- **Don't waste budget on things already tried.** If dropout=0.6 already regressed, don't try it again.
- **Think about the time budget.** Optimise the stage structure for the budget. Deeper stages (layer2, layer1) need lower LRs (~1e-5) and fewer epochs — adding them is only worthwhile if there's enough budget left after layer3 converges.
- **Balance exploitation and exploration.** Fix obvious problems flagged by the analysis, but also try bold moves — combine multiple techniques, try different backbones, or revisit a previously-discarded idea with different parameters. Don't get stuck in a local optimum by only making incremental tweaks.
- **Use a polling interval of 90 seconds.** This is sufficinet and will be up to date enough without wasting tokens.

### Crashes

If a run crashes (OOM, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, revert, and move on.

### NEVER STOP

Once the experiment loop has begun, do NOT pause to ask whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer. You are autonomous. If you run out of ideas, think harder -- re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until you are manually stopped.
