#!/usr/bin/env bash
# Training command v6 — see changes.md for full history

/root/imitationLearning/.venv/bin/python3 train.py \
  --total-steps 10000000 \
  --num-envs 32 \
  --subproc \
  --compile \
  --rollout-steps 2048 \
  --ppo-epochs 4 \
  --minibatch-size 512 \
  --lr 3e-4 \
  --lr-min 1e-5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --vf-coef 1.0 \
  --ent-coef-start 0.01 \
  --ent-coef-end 0.005 \
  --max-grad-norm 5.0 \
  --target-kl 0.015 \
  --threshold 30.0 \
  --window 10 \
  --replay-frac 0.3 \
  --video-interval 25_000 \
  --checkpoint-interval 250_000 \
  --keep-checkpoints 5
