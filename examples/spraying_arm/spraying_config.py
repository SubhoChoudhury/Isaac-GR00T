from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

spraying_config = {
    # Two cameras: base (third-person) and wrist (egocentric)
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["base", "wrist"],
    ),
    # 9-dim state: arm joints (0-7) + sprayer power (8)
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "sprayer"],
    ),
    # 16-step action horizon; dataset stores absolute positions,
    # GR00T processor computes deltas for arm at training time
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "sprayer"],
        action_configs=[
            # arm joints: RELATIVE (delta from current state, better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # sprayer: ABSOLUTE (binary on/off works better as absolute target)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(spraying_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
