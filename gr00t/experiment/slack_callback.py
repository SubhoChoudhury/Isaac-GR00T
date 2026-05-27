import math
import os
import traceback
import requests
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


def _post(text: str):
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        return
    try:
        requests.post(url, json={"text": text}, timeout=5)
    except Exception:
        pass


class SlackNotificationCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        _post(
            f":rocket: *GR00T Run 7 started* — spraying-v7 (sim data) | state-noise-aug + albumentations + checkpoint eval\n"
            f"Target: {args.max_steps:,} steps | Batch: {args.per_device_train_batch_size} | H100 NVL"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0 and state.global_step % 100000 == 0:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            if loss:
                _post(
                    f":bar_chart: Step *{state.global_step:,}* / {args.max_steps:,} "
                    f"({100 * state.global_step / args.max_steps:.0f}%) — "
                    f"loss: {loss:.4f} | lr: {lr:.2e}"
                )

    def on_train_end(self, args, state, control, **kwargs):
        last = state.log_history
        loss = next((e.get("loss") for e in reversed(last) if "loss" in e), None)
        loss_str = f"{loss:.4f}" if loss else "?"
        _post(
            f":white_check_mark: *Training complete* — {state.global_step:,} / {args.max_steps:,} steps | "
            f"final loss: {loss_str}"
        )

    def on_exception(self, args, state, control, exception=None, **kwargs):
        tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        short = tb[-1500:] if len(tb) > 1500 else tb
        _post(f":x: *Training error* at step {state.global_step:,}:\n```{short}```")


class GradExplosionGuardCallback(TrainerCallback):
    """Stop training and notify Slack when gradient norm explodes.

    A single spike above the threshold triggers a Slack warning.
    Three consecutive spikes trigger a clean stop.
    """

    def __init__(self, explosion_factor: float = 20.0, max_consecutive: int = 3):
        # grad_norm must exceed max_grad_norm * explosion_factor to count as explosion
        self._explosion_factor = explosion_factor
        self._max_consecutive = max_consecutive
        self._consecutive = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        grad_norm = logs.get("grad_norm")
        if grad_norm is None:
            return
        # Treat NaN grad_norm as explosion too
        is_nan = isinstance(grad_norm, float) and math.isnan(grad_norm)
        threshold = args.max_grad_norm * self._explosion_factor
        if is_nan or grad_norm > threshold:
            self._consecutive += 1
            norm_str = "NaN" if is_nan else f"{grad_norm:.2f}"
            _post(
                f":boom: *Gradient explosion* at step {state.global_step:,} — "
                f"grad_norm={norm_str} (threshold={threshold:.1f}) | "
                f"consecutive: {self._consecutive}/{self._max_consecutive}"
            )
            if self._consecutive >= self._max_consecutive:
                _post(
                    f":octagonal_sign: *Training stopped* — {self._consecutive} consecutive "
                    f"gradient explosions at step {state.global_step:,}"
                )
                control.should_training_stop = True
        else:
            self._consecutive = 0


# ── Checkpoint eval callback ──────────────────────────────────────────────────

import subprocess
import sys


# ── Checkpoint eval callback ──────────────────────────────────────────────────

import subprocess
import sys
from pathlib import Path


class CheckpointEvalCallback(TrainerCallback):
    """After each checkpoint save, spawn gr00t_checkpoint_eval.py in the background.

    The subprocess loads the saved checkpoint via Gr00tPolicy, runs open-loop
    inference on a fixed eval episode, generates the 9-dim GT vs predicted
    timeline plot, and posts it to Slack. Training is not blocked.

    Requires env vars: SLACK_BOT_TOKEN and SLACK_CHANNEL_ID for image upload.
    """

    def __init__(self, eval_episode_dir: str, max_frames: int = 200):
        self._eval_episode_dir = eval_episode_dir
        self._max_frames       = max_frames
        self._script = Path(__file__).resolve().parent / "gr00t_checkpoint_eval.py"

    def on_save(self, args, state, control, **kwargs):
        step     = state.global_step
        if step % 30000 != 0:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        out_dir  = Path(args.output_dir) / "eval_plots"
        log_path = Path(args.output_dir) / f"eval_step{step:07d}.log"

        if not ckpt_dir.exists():
            _post(f":warning: Eval skipped — checkpoint-{step:,} dir not found at {ckpt_dir}")
            return

        cmd = [
            sys.executable, str(self._script),
            "--checkpoint-dir",   str(ckpt_dir),
            "--eval-episode-dir", self._eval_episode_dir,
            "--step",             str(step),
            "--max-frames",       str(self._max_frames),
            "--output-dir",       str(out_dir),
        ]

        log_f = open(log_path, "w")
        subprocess.Popen(cmd, env=os.environ.copy(), stdout=log_f, stderr=subprocess.STDOUT)
        _post(f":hourglass_flowing_sand: *Eval launched* for checkpoint-{step:,} — plot on Slack in ~90s")
