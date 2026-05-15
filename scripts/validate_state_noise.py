"""
Validate the state-noise augmentation added to Gr00tN1d7Processor.

Loads the local checkpoint, instantiates the processor with state_noise_std=0.01
and state_noise_keys=("arm",), then verifies:

  1. With self.training=True, the model-visible arm state changes between calls
     while the sprayer state is byte-identical (sprayer not in state_noise_keys).
  2. The empirical noise (visible_arm - clean_arm) over many samples has
     mean ~= 0 and std ~= 0.01.
  3. With self.training=False (eval), there is no noise -> deterministic.
  4. The action-target tensor produced by the processor is identical across
     noisy runs (proves the relative-action conversion was computed from the
     clean raw state and is therefore not poisoned by the noise injection).

Run:
    cd /mnt/ssd2tb/Isaac-GR00T
    python scripts/validate_state_noise.py \\
        --checkpoint /mnt/ssd2tb/gr00t/checkpoint-22500 \\
        --sigma 0.01 --num-samples 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Importing the gr00t model package registers Gr00tN1d7Processor with AutoProcessor.
import gr00t.model.gr00t_n1d7  # noqa: F401
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


def build_dummy_step(processor) -> VLAStepData:
    """Construct a minimal VLAStepData matching the embodiment's modality config."""
    embodiment_tag = "new_embodiment"
    modality_cfg = processor.modality_configs[embodiment_tag]

    # State: pick one timestep per state key with the right joint dim.
    states: dict[str, np.ndarray] = {}
    for key in modality_cfg["state"].modality_keys:
        dim = processor.state_action_processor.norm_params[embodiment_tag]["state"][key][
            "dim"
        ].item()
        states[key] = np.zeros((1, dim), dtype=np.float32)

    # Actions: action_horizon steps per action key with the right action dim.
    actions: dict[str, np.ndarray] = {}
    action_horizon = len(modality_cfg["action"].delta_indices)
    for key in modality_cfg["action"].modality_keys:
        dim = processor.state_action_processor.norm_params[embodiment_tag]["action"][key][
            "dim"
        ].item()
        actions[key] = np.zeros((action_horizon, dim), dtype=np.float32)

    # Dummy images: one black PIL image per camera view.
    from PIL import Image as PILImage

    images: dict[str, list] = {}
    for key in modality_cfg["video"].modality_keys:
        images[key] = [PILImage.new("RGB", (224, 224), color=0)]

    return VLAStepData(
        images=images,
        masks=None,
        states=states,
        actions=actions,
        text="spray the surface",
        embodiment=EmbodimentTag(embodiment_tag),
    )


def run_processor(processor, step: VLAStepData) -> dict:
    msg = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
    return processor(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--num-samples", type=int, default=5000)
    args = parser.parse_args()

    print(f"Loading processor from {args.checkpoint} ...")
    processor = Gr00tN1d7Processor.from_pretrained(
        str(args.checkpoint),
        state_noise_std=args.sigma,
        state_noise_keys=("arm",),
        state_dropout_prob=0.0,  # disable dropout for this validation
    )
    print(f"  state_noise_std = {processor.state_noise_std}")
    print(f"  state_noise_keys = {processor.state_noise_keys}")
    print(f"  state_dropout_prob = {processor.state_dropout_prob}")

    # Same dummy step reused every call so the only source of variation is the noise.
    step = build_dummy_step(processor)

    # --- Test 1: train mode -> arm varies, sprayer is identical ---
    processor.train()
    out_a = run_processor(processor, step)
    out_b = run_processor(processor, step)

    # The processor pads state to max_state_dim; the first 8 dims are arm, next 1 is sprayer
    # (per spraying_arm modality config: arm 8D, sprayer 1D).
    state_a = out_a["state"].cpu().numpy()
    state_b = out_b["state"].cpu().numpy()
    arm_a, arm_b = state_a[:, :8], state_b[:, :8]
    spr_a, spr_b = state_a[:, 8:9], state_b[:, 8:9]

    arm_diff = np.abs(arm_a - arm_b).max()
    spr_diff = np.abs(spr_a - spr_b).max()
    print(f"\n[Test 1] arm max-abs diff between two train calls = {arm_diff:.5f}")
    print(f"[Test 1] sprayer max-abs diff between two train calls = {spr_diff:.5g}")
    assert arm_diff > 0.0, "FAIL: arm state did not change between training calls (no noise applied)"
    assert spr_diff == 0.0, "FAIL: sprayer state changed but should be excluded from noise"
    print("[Test 1] PASS: arm changes; sprayer is byte-identical.")

    # --- Test 2: empirical noise stats ~ N(0, sigma) ---
    np.random.seed(0)
    # Get the clean baseline once in eval mode.
    processor.eval()
    clean_state = run_processor(processor, step)["state"].cpu().numpy()[:, :8].copy()

    processor.train()
    deltas = np.empty((args.num_samples, 8), dtype=np.float32)
    for i in range(args.num_samples):
        deltas[i] = run_processor(processor, step)["state"].cpu().numpy()[0, :8] - clean_state[0]
    emp_mean = float(deltas.mean())
    emp_std = float(deltas.std())
    per_dim_std = deltas.std(axis=0)
    print(
        f"\n[Test 2] over {args.num_samples} samples: noise mean={emp_mean:+.5f}, "
        f"noise std={emp_std:.5f} (target: 0.0 / {args.sigma})"
    )
    print(f"[Test 2] per-joint std: {np.array2string(per_dim_std, precision=4)}")
    assert abs(emp_mean) < args.sigma * 0.1, "FAIL: empirical mean too far from 0"
    assert abs(emp_std - args.sigma) < args.sigma * 0.1, "FAIL: empirical std too far from sigma"
    print("[Test 2] PASS: empirical N(0, sigma) holds within 10%.")

    # --- Test 3: eval mode is deterministic ---
    processor.eval()
    eval_a = run_processor(processor, step)["state"].cpu().numpy()
    eval_b = run_processor(processor, step)["state"].cpu().numpy()
    eval_diff = np.abs(eval_a - eval_b).max()
    print(f"\n[Test 3] eval-mode max-abs state diff between calls = {eval_diff:.5g}")
    assert eval_diff == 0.0, "FAIL: state changed at eval (inference) time — noise must be off"
    print("[Test 3] PASS: eval mode is deterministic.")

    # --- Test 4: action targets are not corrupted by state noise ---
    processor.train()
    act_a = run_processor(processor, step)["action"].cpu().numpy()
    act_b = run_processor(processor, step)["action"].cpu().numpy()
    act_diff = np.abs(act_a - act_b).max()
    print(f"\n[Test 4] train-mode max-abs action-target diff between calls = {act_diff:.5g}")
    assert act_diff == 0.0, (
        "FAIL: action targets differ across noisy training calls — relative-action conversion "
        "may have been computed from a noised state."
    )
    print("[Test 4] PASS: action targets unaffected by state noise.")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
