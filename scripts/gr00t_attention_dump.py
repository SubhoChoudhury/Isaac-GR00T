#!/usr/bin/env python3
"""
gr00t_attention_dump.py

Runs GR00T inference on dataset episodes and saves attention weights
in VLAExplain-compatible format for cross-modal attention visualization.

Usage:
    cd ~/Isaac-GR00T && source .venv/bin/activate
    python scripts/gr00t_attention_dump.py \
        --model-path /persistent/checkpoints/spraying-9dim-run2/spraying-9dim-run2/checkpoint-50000 \
        --dataset-path /data/lerobot/vla/spraying-9dim \
        --output-dir /data/vlaexplain_data \
        --modality-config-path examples/spraying_arm/spraying_config.py \
        --traj-ids 0 1 2 \
        --steps 20
"""

import importlib
import logging
import os
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from PIL import Image
from transformers import BatchFeature

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


# ── Attention capture ──────────────────────────────────────────────────────────

_attn_cache: dict = {}


def _patch_backbone_for_attn_capture(policy: Gr00tPolicy) -> None:
    """Monkey-patch Qwen3Backbone.forward to also capture attention weights."""
    backbone = policy.model.backbone
    backbone.model.set_attn_implementation("eager")  # flash_attn blocks output_attentions

    def capturing_forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_inputs = {k: vl_input[k] for k in keys_to_use if k in vl_input}
        with torch.no_grad():
            outputs = self.model(
                **vl_inputs,
                output_hidden_states=True,
                output_attentions=True,
            )
        _attn_cache.clear()
        _attn_cache["attentions"] = [a.detach().cpu().float() for a in outputs.attentions]
        _attn_cache["input_ids"] = vl_inputs["input_ids"].detach().cpu()
        hidden = outputs.hidden_states[-1]
        image_mask = vl_inputs["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_inputs["attention_mask"] == 1
        return BatchFeature(data={
            "backbone_features": hidden,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        })

    backbone.__class__.forward = capturing_forward
    logging.info("Patched Qwen3Backbone.forward for attention capture.")


# ── Sequence layout ────────────────────────────────────────────────────────────

def compute_segment_indices(input_ids: torch.Tensor, image_token_id: int) -> dict:
    ids = input_ids[0].tolist()
    image_runs = []
    in_run, run_start = False, 0
    for i, tok in enumerate(ids):
        if tok == image_token_id:
            if not in_run:
                in_run, run_start = True, i
        elif in_run:
            image_runs.append((run_start, i))
            in_run = False
    if in_run:
        image_runs.append((run_start, len(ids)))
    segs = {}
    if len(image_runs) >= 1:
        segs["image1"] = image_runs[0]
    if len(image_runs) >= 2:
        segs["image2"] = image_runs[1]
    text_start = image_runs[-1][1] if image_runs else 0
    segs["text"] = (text_start, len(ids))
    return segs


# ── VLAExplain format writers ──────────────────────────────────────────────────

def save_attention_step(attn_dir: Path, step: int, attentions: list) -> None:
    data = {step: {i: a[0] for i, a in enumerate(attentions)}}
    with open(attn_dir / f"{step}_expert_attention.pkl", "wb") as f:
        pickle.dump(data, f)


def save_language_info(lang_dir: Path, info: dict) -> None:
    with open(lang_dir / "language_info.pkl", "wb") as f:
        pickle.dump(info, f)


def save_raw_images(img_dir: Path, step: int, obs: dict) -> None:
    for cam_idx, cam_key in enumerate(["base", "wrist"], start=1):
        arr = obs["video"][cam_key]
        while arr.ndim > 3:
            arr = arr[0]
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(img_dir / f"step_{step:04d}_image{cam_idx}.jpg")


def parse_observation_gr00t(obs: dict, modality_configs: dict) -> dict:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs[parsed_key]
            new_obs[modality][key] = [[arr]] if isinstance(arr, str) else arr[None, :]
    return new_obs


# ── Main ───────────────────────────────────────────────────────────────────────

@dataclass
class DumpConfig:
    model_path: str
    """Path to GR00T checkpoint directory (e.g. .../checkpoint-50000)."""

    dataset_path: str = "/data/lerobot/vla/spraying-9dim"
    embodiment_tag: str = "new_embodiment"
    modality_config_path: str = "examples/spraying_arm/spraying_config.py"
    output_dir: str = "/data/vlaexplain_data"
    traj_ids: list[int] = field(default_factory=lambda: [0])
    steps: int = 20
    """Number of inference steps per episode (one step = one forward pass = 16-action chunk)."""
    action_horizon: int = 16


def main(cfg: DumpConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    path = Path(cfg.modality_config_path)
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    cfg.embodiment_tag = EmbodimentTag.resolve(cfg.embodiment_tag)

    out = Path(cfg.output_dir)
    attn_dir, lang_dir, img_dir = out / "expert_attention", out / "language_info", out / "raw_images"
    for d in [attn_dir, lang_dir, img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading GR00T from {cfg.model_path} ...")
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=cfg.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy.model.eval()
    _patch_backbone_for_attn_capture(policy)

    modality = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=cfg.dataset_path,
        modality_configs=modality,
        video_backend="av",
    )
    image_token_id = policy.model.backbone.model.config.image_token_id

    language_info: dict = {}
    global_step = 0

    for traj_id in cfg.traj_ids:
        traj = dataset[traj_id]
        actual_steps = min(cfg.steps * cfg.action_horizon, len(traj))
        mc = deepcopy(modality)
        mc.pop("action")

        for step_offset in range(0, actual_steps, cfg.action_horizon):
            data_point = extract_step_data(traj, step_offset, mc, cfg.embodiment_tag)
            obs = {}
            for k, v in data_point.states.items():
                obs[f"state.{k}"] = v
            for k, v in data_point.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lang_key in modality["language"].modality_keys:
                obs[lang_key] = data_point.text
            parsed_obs = parse_observation_gr00t(obs, modality)

            with torch.no_grad():
                policy.get_action(parsed_obs)

            if not _attn_cache:
                logging.warning(f"No attention at step {global_step}, skipping.")
                global_step += 1
                continue

            attentions = _attn_cache["attentions"]
            input_ids = _attn_cache["input_ids"]

            save_attention_step(attn_dir, global_step, attentions)
            save_raw_images(img_dir, global_step, parsed_obs)

            state_flat = np.concatenate([
                data_point.states[k].flatten() for k in modality["state"].modality_keys
            ])
            language_info[global_step] = {
                "text_token_ids": input_ids.cpu(),
                "state": torch.tensor(state_flat, dtype=torch.float32).unsqueeze(0),
            }
            save_language_info(lang_dir, language_info)

            if global_step == 0:
                segs = compute_segment_indices(input_ids, image_token_id)
                n_layers = len(attentions)
                n_heads = attentions[0].shape[1]
                seq_len = attentions[0].shape[-1]
                logging.info(f"Sequence layout: {segs}")
                logging.info(f"Layers={n_layers}, heads={n_heads}, seq_len={seq_len}")
                with open(out / "segment_info.txt", "w") as f:
                    for k, v in segs.items():
                        f.write(f"{k}: {v}\n")
                    f.write(f"num_layers: {n_layers}\n")
                    f.write(f"num_heads: {n_heads}\n")
                    f.write(f"seq_len: {seq_len}\n")
                    patches_per_cam = segs.get("image1", (0, 0))
                    grid = int((patches_per_cam[1] - patches_per_cam[0]) ** 0.5)
                    f.write(f"patch_grid: {grid}x{grid}\n")

            logging.info(f"Saved step {global_step} (traj {traj_id}, offset {step_offset})")
            global_step += 1

    logging.info(f"Done. {global_step} steps written to {out}")
    logging.info(
        f"\nTo launch VLAExplain UI:\n"
        f"  LEROBOT_DATA_DIR={out} "
        f"TOKENIZER_PATH=nvidia/Cosmos-Reason2-2B "
        f"LAN_MODEL_LAYER_NUM=16 "
        f"IGNORE_IMAGE3=True "
        f"python ~/VLAExplain/src/lerobot/main.py"
    )


if __name__ == "__main__":
    main(tyro.cli(DumpConfig))
