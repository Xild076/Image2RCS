# Image2RCS

Image2RCS trains a computer vision regression model to estimate aircraft radar cross section from images.

**First of all:** Why?
**Answer:** Idk, it seemed fun. I like: planes and AI, might as well combine the two? Oh well. I plan to keep adding data, note that NONE of these images are mine.

## Project Structure

- src/dataset.py: dataset loading, image transforms, target transforms
- src/model.py: regression model definition
- src/train.py: full training loop, checkpointing, and profiling
- src/inference.py: checkpoint loading, batched inference CLI, and profiling
- src/benchmark.py: convenience benchmark wrapper for train/inference profiling
- data/aircraft_rcs.csv: RCS labels per aircraft class or folder
- data/images/: image folders used for training and inference

## Install

Recommended CPU runtime (matches project virtualenv):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision pandas pillow numpy
```

Python 3.12 is recommended for stable PyTorch CPU behavior.

## Train

```bash
python src/train.py \
  --csv-path data/aircraft_rcs.csv \
  --images-root data/images \
  --output checkpoints/best_model.pt \
  --epochs 20 \
  --batch-size 16 \
  --num-workers auto \
  --cache-mode disk \
  --cache-dir .cache/image2rcs \
  --eval-every 1 \
  --profile
```

Useful options:

- `--num-workers auto|N`: auto-tune DataLoader workers over `{0,1,2,4}`
- `--cache-mode off|memory|disk`: image decode cache strategy (`disk` is fastest for repeat runs)
- `--memory-cache-items N`: cap RAM usage when `--cache-mode memory` (default `256`)
- `--cache-dir PATH`: cache directory for disk mode
- `--persistent-workers`: keep workers across epochs (faster, higher RAM)
- `--prefetch-factor N`: prefetched batches per worker (default `1`, lower RAM)
- `--cpu-threads N`: set Torch CPU thread count
- `--eval-every N`: run validation every N epochs
- `--profile`: print per-epoch timing split (`data_load`, `fwd_bwd`, `eval`, `ckpt_io`)
- `--device auto|gpu|cuda|cuda:N|mps|cpu`: select compute backend (`auto` prefers CUDA, then Apple MPS, then CPU)

## Inference

Single image:

```bash
python src/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --input data/images/f-22_images/1.png \
  --batch-size 1 \
  --num-workers auto \
  --cache-mode disk \
  --profile
```

Directory:

```bash
python src/inference.py \
  --checkpoint checkpoints/best_model.pt \
  --input data/images/f-22_images \
  --batch-size 16 \
  --num-workers auto \
  --cache-mode disk \
  --profile
```

Useful options:

- `--batch-size N`: batched folder inference for much higher throughput
- `--num-workers auto|N`: auto-tune inference DataLoader workers
- `--cache-mode off|memory|disk`: image decode cache strategy
- `--memory-cache-items N`: cap RAM usage when `--cache-mode memory` (default `256`)
- `--cache-dir PATH`: cache directory for disk mode
- `--cpu-threads N`: set Torch CPU thread count
- `--profile`: print `images/sec` and timing split
- `--device auto|gpu|cuda|cuda:N|mps|cpu`: select compute backend (`auto` prefers CUDA, then Apple MPS, then CPU)

## Benchmark

```bash
python src/benchmark.py --train --infer --train-epochs 2 --batch-size 16 --cache-mode disk --num-workers auto
```

## Render Photos (With Approval)

Generate dense synthetic photos from meshes in `data/renders`:

```bash
python src/render_photos.py \
  --render-root data/renders \
  --images-root data/images \
  --approval-mode interactive \
  --clean-output
```

Approval records and review sheets are written under `data/model_approvals`.

- `--approval-mode interactive`: prompt approve/reject per model after generating a review sheet
- `--approval-mode check`: render only previously approved models
- `--approval-mode only`: review and approve/reject only, without rendering dataset images
- `--approval-mode off`: skip approval gating entirely
- `--auto-approve-all-first`: pre-approve every discovered model once at startup

Quality/detail controls:

- `--render-size`: output image size (default `896`)
- `--preview-size`: approval-sheet render size (default `512`)
- `--azimuth-step`: smaller means more camera angles (default `20`)
- `--elevation-min/--elevation-max/--elevation-step`: vertical angle sweep (default `-30..30` by `15`)
- `--gc-every-frames`: periodic garbage collection cadence during render (default `40`)

Lighting and sky variation:

- Every render cycles through sky backgrounds: `lightblue`, `duskorange`, and `black`
- Lighting cycles through multiple positions: `left_high`, `right_high`, `front_low`, `back_rim`
- Variants are included in output filenames for easier filtering and debugging

Useful for quick validation:

```bash
python src/render_photos.py \
  --limit-models 1 \
  --approval-mode off \
  --render-size 256 \
  --azimuth-step 90 \
  --elevation-min -20 \
  --elevation-max 20 \
  --elevation-step 20
```

This is a really basic image regression model that takes any image input and outputs an estimated radar cross section, naturally, the model isn't very good due to very limited data... but hey, it's cool!

Btw: credits due to the creators of the 3d models for these planes... I found them on sketchfab. I will include more detailed cites later!
