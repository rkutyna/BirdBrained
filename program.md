# Autoresearch: Bird Species Classifier

## What this is

A ResNet-50 fine-grained bird species classifier trained on the NABirds dataset (98 target species, ~10.5k train / ~1.2k val images). Images are cropped to bounding boxes, resized/padded to 240x240, and normalised with ImageNet statistics.

The training uses a progressive unfreezing strategy:
1. **Stage 1** -- train the FC head only (backbone frozen)
2. **Stage 2** -- unfreeze `layer4` + FC
3. **Stage 3** -- unfreeze `layer3` + `layer4` + FC

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

- **Augmentation pipeline**: `CROP_SCALE_MIN/MAX`, `JITTER_*`, `RANDOM_ERASING_*`, or add new transforms in `build_transforms()`
- **Learning rates**: per-stage `lr` values in the `STAGES` list
- **Stage structure**: number of stages, which layers to unfreeze, max epochs per stage
- **Dropout rate**: `DROPOUT`
- **Backbone**: `BACKBONE` -- swap to `"efficientnet_b0"` or `"mobilenet_v3_large"` (support is already wired in)
- **Label smoothing**: `LABEL_SMOOTHING`
- **Optimizer**: `OPTIMIZER` -- `"adam"`, `"adamw"`, or `"sgd"` (with `MOMENTUM`)
- **Scheduler**: `SCHEDULER` -- `"none"`, `"cosine"`, or `"step"`
- **Batch size**: `BATCH_SIZE`
- **Mixed precision**: `USE_AMP`
- **Time budget**: `TIME_BUDGET_SEC` (only if explicitly instructed)
- **NOTES**: always update this to describe what you changed

You may also modify `build_transforms()`, `build_model()`, `build_optimizer()`, `build_scheduler()`, and the training loop functions for more structural changes (e.g. adding mixup, changing the head architecture, adding gradient clipping).

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
```

You can extract the key metric:

```
grep "^top1_val_acc:" run.log
```

## The experiment loop

LOOP FOREVER:

1. **Read** `artifacts/autoresearch_log.csv` to understand the current best accuracy and what has been tried.
2. **Propose** one change with a hypothesis (e.g. "adding cosine annealing should help the later stages converge better").
3. **Edit** `train.py` -- make the change and update `NOTES` to describe it.
4. **Run** `python train.py` and wait for it to complete.
5. **Read** the output to see the result (check the `---` summary block or the last line of `artifacts/autoresearch_log.csv`).
6. **Decide**: if `top1_val_acc` improved, keep the change. If it regressed, revert `train.py` to the previous version.
7. **Repeat** from step 1.

### Guidelines

- **One change at a time.** Changing multiple things makes it impossible to attribute improvement or regression.
- **Keep NOTES concise and descriptive.** Good: `"adamw lr=3e-4, cosine schedule"`. Bad: `"trying something new"`.
- **Read the log before every run.** The history tells you what has and hasn't worked.
- **Don't waste budget on things already tried.** If dropout=0.6 already regressed, don't try it again.
- **Think about the time budget.** With 30 minutes, you have room for all 3 stages and some extra depth, but not unlimited epochs. Optimise the stage structure for the budget.
- **Low-hanging fruit first.** Try optimizer/LR/scheduler changes before architectural changes.

### Crashes

If a run crashes (OOM, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, revert, and move on.

### NEVER STOP

Once the experiment loop has begun, do NOT pause to ask whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer. You are autonomous. If you run out of ideas, think harder -- re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until you are manually stopped.
