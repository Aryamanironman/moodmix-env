# Running MoodMix on AWS

The environment is feature-based, so training is **CPU-bound**. A GPU is only
useful if you enlarge the network or add an audio model; a tiny 128×128 MLP
usually trains faster on a fast CPU than on a GPU.

## Sizing

| target | laptop (12 threads) | `c6i.4xlarge` (16 vCPU) | `c6i.8xlarge` (32 vCPU) |
|--------|---------------------|--------------------------|--------------------------|
| 100k steps | ~6 min | ~3 min | ~2 min |
| 1M steps | ~50-65 min | ~25-40 min | ~15-25 min |
| 10M steps | ~9-11 h | ~4-7 h | ~2.5-4 h |
| 100M steps | ~4-5 days | ~2 days | ~1-1.5 days |

100M is not recommended for this environment. The policy is small and the
reward plateaus long before then; spend the time on a real crate, a continuous
action space, or hyperparameter sweeps instead.

## Setup (Ubuntu EC2)

```bash
ssh ubuntu@<instance>
sudo apt update && sudo apt install -y python3-venv git tmux awscli
git clone <your-repo> moodmix-env      # or rsync the project
cd moodmix-env
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
python scripts/make_demo_library.py    # or rsync data/demo from your laptop
```

Run inside `tmux` so an SSH disconnect does not kill training:

```bash
tmux new -s moodmix
```

## 1M-step run (fresh)

```bash
export BUCKET=my-moodmix-bucket

python scripts/train.py \
  --timesteps 1000000 \
  --n-envs 16 \
  --vec-env subproc \
  --n-steps 256 \
  --batch-size 512 \
  --checkpoint-freq 50000 \
  --checkpoint-dir /data/checkpoints \
  --output /data/models/ppo_1m \
  --s3-uri "s3://$BUCKET/moodmix" \
  --eval-episodes 20 \
  --verbose 0
```

`--s3-uri` uploads every checkpoint and the final model. The instance role
needs `s3:PutObject` (and `s3:ListBucket` if you list the prefix).

## Resume

`--timesteps` means **additional** steps, not total.

```bash
python scripts/train.py \
  --resume /data/checkpoints/ppo_moodmix_500000_steps.zip \
  --timesteps 500000 \
  --learning-rate 1e-4 \
  --n-envs 16 --vec-env subproc \
  --checkpoint-freq 50000 \
  --checkpoint-dir /data/checkpoints \
  --output /data/models/ppo_1m \
  --s3-uri "s3://$BUCKET/moodmix" \
  --verbose 0
```

For spot instances: download the newest `ppo_moodmix_*_steps.zip` from S3 and
resume with the remaining budget. Checkpoints are independent of the instance.

## TensorBoard

```bash
tensorboard --logdir /data/tensorboard --port 6006 --bind_all
```

Or tunnel it:

```bash
ssh -L 6006:localhost:6006 ubuntu@<instance>
# open http://localhost:6006
```

## Evaluate

```bash
python scripts/evaluate.py \
  --model /data/models/ppo_1m.zip \
  --episodes 20
```

The table compares PPO, the hand-written human baseline, and random on identical
seeds. `completed` is the percentage of transitions that finish on deck B;
`curve error` is the mean distance from the intended fader curve.

## Real music crate

```bash
pip install -e ".[library,spotify]"
python scripts/analyze_library.py --audio-dir /data/crate
# or, for a Spotify playlist matched to local files:
python scripts/import_spotify.py --playlist <url> --audio-dir /data/crate
```

Then add `--data-dir /data/crate` to training and evaluation.

## Web demo on the instance

```bash
python scripts/serve_demo.py \
  --model /data/models/ppo_1m.zip \
  --host 0.0.0.0 --port 8050
```

Open port 8050 in the security group, or tunnel it. This is Flask's development
server; do not expose it directly to the public internet without a reverse proxy
and auth.

## GPU note

If you have a CUDA-capable instance and a CUDA build of PyTorch:

```bash
python scripts/train.py ... --device cuda
```

`--device auto` selects CUDA when available. Expect little speedup for the
current 64-observation, 10-control environment because stepping is done on the
CPU. GPU pays off when the policy or audio model becomes large.
