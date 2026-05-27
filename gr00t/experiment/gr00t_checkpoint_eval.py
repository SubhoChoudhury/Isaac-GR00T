#!/usr/bin/env python3
"""Standalone open-loop eval: GR00T checkpoint vs GT on a fixed pi05-format episode.

Generates the 9-dim GT vs predicted timeline plot and posts it to Slack.

Usage:
    python gr00t_checkpoint_eval.py \
        --checkpoint-dir /data/checkpoints/spraying-v7-run7/checkpoint-10000 \
        --eval-episode-dir /data/sim-cleaned/.../run_20260502_033038_d51d_job_000 \
        --step 10000 \
        --max-frames 200 \
        --output-dir /data/checkpoints/spraying-v7-run7/eval_plots
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio

import matplotlib
matplotlib.use("Agg")  # non-interactive — no display needed
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

# ── Robot constants (mirror spraying_config.py) ───────────────────────────────
STATE_INDICES  = [0, 4, 8, 9, 10, 11, 12, 13]
TOOL_PWR_INDEX = 3
IMAGE_RESOLUTION = (224, 224)
DIM_NAMES = [
    "lift_mid", "lift_top",
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
    "sprayer_pwr",
]
PROMPT = (
    "Spray the unsprayed regions of the wall with L5 spray to achieve full, "
    "even coverage across the entire wall surface"
)

# ── Helpers (identical to eval notebooks) ────────────────────────────────────

def _slice_state_9d(raw_state, raw_tool):
    arm  = np.asarray(raw_state, dtype=np.float32)[STATE_INDICES]
    tool = np.asarray(raw_tool,  dtype=np.float32)[TOOL_PWR_INDEX:TOOL_PWR_INDEX + 1]
    return np.concatenate([arm, tool], axis=0)


def _build_gt_actions(states_9d: np.ndarray) -> np.ndarray:
    """pi5-format GT: delta for arm joints, absolute next-step for sprayer."""
    delta    = states_9d[1:, :8] - states_9d[:-1, :8]
    tool_nxt = states_9d[1:, 8:9]
    return np.concatenate([delta, tool_nxt], axis=1).astype(np.float32)


def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = IMAGE_RESOLUTION
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _load_episode(episode_dir: Path, ep_idx: int):
    stem   = f"file-{ep_idx:03d}"
    pq_p   = episode_dir / "data"   / "chunk-000" / f"{stem}.parquet"
    base_p = episode_dir / "videos" / "observation.images.base"  / "chunk-000" / f"{stem}.mp4"
    wrst_p = episode_dir / "videos" / "observation.images.wrist" / "chunk-000" / f"{stem}.mp4"

    for p in (pq_p, base_p, wrst_p):
        if not p.exists():
            raise FileNotFoundError(p)

    df = pd.read_parquet(pq_p).sort_values("frame_index").reset_index(drop=True)
    n  = len(df)
    raw_s = np.stack(df["observation.state"].values).astype(np.float32)
    raw_t = np.stack(df["observation.tool_state"].values).astype(np.float32)
    states_9d = np.stack([_slice_state_9d(raw_s[i], raw_t[i]) for i in range(n)])
    gt_acts   = _build_gt_actions(states_9d)

    base_frames  = np.stack([f for f in iio.imiter(str(base_p),  plugin="pyav")])
    wrist_frames = np.stack([f for f in iio.imiter(str(wrst_p),  plugin="pyav")])
    return states_9d, gt_acts, base_frames, wrist_frames


def _build_obs(state_9d, base_img, wrist_img):
    arm     = state_9d[:8].astype(np.float32).reshape(1, 1, 8)
    sprayer = state_9d[8:9].astype(np.float32).reshape(1, 1, 1)
    base_b  = base_img.astype(np.uint8).reshape(1, 1, *base_img.shape)
    wrist_b = wrist_img.astype(np.uint8).reshape(1, 1, *wrist_img.shape)
    return {
        "video":    {"base": base_b, "wrist": wrist_b},
        "state":    {"arm": arm, "sprayer": sprayer},
        "language": {
            "task":                              [[PROMPT]],
            "annotation.human.task_description": [[PROMPT]],
        },
    }


def _pred_to_delta(action_dict, state_9d_t):
    arm_abs  = np.asarray(action_dict["arm"])[0, 0]
    sprayer  = np.asarray(action_dict["sprayer"])[0, 0]
    arm_delta = arm_abs - state_9d_t[:8]
    return np.concatenate([arm_delta, sprayer], axis=0).astype(np.float32)


def _find_episode_indices(episode_dir: Path) -> list[int]:
    d = episode_dir / "data" / "chunk-000"
    if not d.exists():
        return []
    return sorted(int(p.stem.split("-")[1]) for p in d.glob("file-*.parquet"))


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_timeline(pred: np.ndarray, gt: np.ndarray, step: int,
                   ep_label: str, out_path: Path) -> None:
    t = np.arange(len(gt))
    fig, axes = plt.subplots(
        len(DIM_NAMES), 1,
        figsize=(13, 2.2 * len(DIM_NAMES)),
        sharex=True, constrained_layout=True,
    )
    for i, d in enumerate(range(9)):
        axes[i].plot(t, gt[:,   d], "k-",  linewidth=1.0, alpha=0.7, label="GT")
        axes[i].plot(t, pred[:, d], "C0-", linewidth=1.0, alpha=0.7, label=f"ckpt-{step:,}")
        axes[i].set_ylabel(DIM_NAMES[d], fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("frame index")
    axes[0].set_title(f"checkpoint-{step:,}  |  {ep_label}", fontsize=10)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Plot saved → {out_path}")


# ── Slack ─────────────────────────────────────────────────────────────────────

def _post_image(image_path: Path, comment: str) -> None:
    token   = os.environ.get("SLACK_BOT_TOKEN",  "")
    channel = os.environ.get("SLACK_CHANNEL_ID", "")
    if not token or not channel:
        print("[eval] SLACK_BOT_TOKEN / SLACK_CHANNEL_ID not set — skipping Slack upload")
        return
    try:
        from slack_sdk import WebClient
        client = WebClient(token=token)
        client.files_upload_v2(
            channel=channel,
            file=str(image_path),
            title=image_path.name,
            initial_comment=comment,
        )
        print("[eval] Posted to Slack ✓")
    except Exception as e:
        print(f"[eval] Slack upload failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir",   required=True, help="Path to saved HF checkpoint dir")
    ap.add_argument("--eval-episode-dir", required=True, help="pi05 session dir with data/ + videos/")
    ap.add_argument("--step",  type=int, default=0,   help="Training step (for labelling)")
    ap.add_argument("--max-frames", type=int, default=200, help="Frames to evaluate per episode")
    ap.add_argument("--output-dir", default="/tmp/gr00t_eval", help="Where to save the PNG")
    args = ap.parse_args()

    ckpt_dir   = Path(args.checkpoint_dir)
    ep_dir     = Path(args.eval_episode_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure gr00t package is importable when run as subprocess
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # ── Load policy ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Loading policy from {ckpt_dir} on {device}…")
    t0 = time.monotonic()
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=str(ckpt_dir),
        device=device,
    )
    print(f"[eval] Policy loaded in {time.monotonic()-t0:.1f}s")

    # ── Pick episode ──────────────────────────────────────────────────────────
    indices = _find_episode_indices(ep_dir)
    if not indices:
        print(f"[eval] No episodes found in {ep_dir}")
        return
    ep_idx = indices[0]
    print(f"[eval] Episode: {ep_dir.name}/file-{ep_idx:03d}")

    states_9d, gt_actions, base_frames, wrist_frames = _load_episode(ep_dir, ep_idx)
    n_eval = min(args.max_frames, len(gt_actions))
    print(f"[eval] Evaluating {n_eval} frames…")

    # ── Inference loop ────────────────────────────────────────────────────────
    pred = np.zeros((n_eval, 9), dtype=np.float32)
    lats = []
    for t in range(n_eval):
        obs = _build_obs(states_9d[t], _resize(base_frames[t]), _resize(wrist_frames[t]))
        t1  = time.monotonic()
        action_dict, _ = policy.get_action(obs)
        lats.append((time.monotonic() - t1) * 1000)
        pred[t] = _pred_to_delta(action_dict, states_9d[t])
        if (t + 1) % 50 == 0:
            print(f"  {t+1}/{n_eval}  lat_p50={np.median(lats):.0f}ms")

    gt = gt_actions[:n_eval]

    # ── Stats ─────────────────────────────────────────────────────────────────
    mae = np.abs(pred - gt).mean(axis=0)
    overall_mae = float(mae.mean())
    cos_sims = np.array([
        float((pred[:, d] * gt[:, d]).sum() /
              (np.linalg.norm(pred[:, d]) * np.linalg.norm(gt[:, d]) + 1e-9))
        for d in range(9)
    ])
    print(f"[eval] Overall MAE : {overall_mae:.4f}")
    print(f"[eval] Cos sim     : {np.array2string(cos_sims, precision=3)}")
    print(f"[eval] Lat p50/p95 : {np.median(lats):.0f} / {np.percentile(lats, 95):.0f} ms")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_png = out_dir / f"eval_step{args.step:07d}.png"
    ep_label = f"{ep_dir.parent.name}/{ep_dir.name}/file-{ep_idx:03d}"
    _plot_timeline(pred, gt, args.step, ep_label, out_png)

    # ── Slack ─────────────────────────────────────────────────────────────────
    dim_lines = "  ".join(
        f"{DIM_NAMES[d]}:{cos_sims[d]:+.2f}" for d in range(9)
    )
    comment = (
        f":microscope: *Open-loop eval — checkpoint-{args.step:,}*  `#gr00t`\n"
        f"MAE: `{overall_mae:.4f}`  |  lat p50: `{np.median(lats):.0f}ms`\n"
        f"cos_sim per dim:  `{dim_lines}`"
    )
    _post_image(out_png, comment)


if __name__ == "__main__":
    main()
