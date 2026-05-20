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
            f":rocket: *GR00T Run 3 started* — undistorted base cam | state-noise-aug\n"
            f"Target: {args.max_steps:,} steps | Batch: {args.per_device_train_batch_size} | H100 NVL"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0 and state.global_step % 15000 == 0:
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
