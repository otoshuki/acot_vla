"""See _CONFIGS for the list of available configs."""
from __future__ import annotations
import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias, Dict, Union
import os
import numpy as np
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.acot_vla as acot_vla
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.agilex_policy as agilex_policy
import openpi.policies.go1_policy as go1_policy
import openpi.policies.go2_policy as go2_policy
import openpi.policies.vlabench_policy as vlabench_policy
import openpi.policies.arx_policy as arx_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.models.acot_multilora as acot_multilora

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    prompt_from_hl_instruction: bool = False

    dataloader_sampler: str | None = ''

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.ACOT_VLA_PI0:
                assert isinstance(model_config, acot_vla.ACOTConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.ACOTPadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.ACOT_VLA_PI05:
                assert isinstance(model_config, acot_vla.ACOTConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.ACOTPadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            if not isinstance(asset_id, list):
                asset_id = [asset_id]

            # mean of those norm stats:
            all_norm_stats = []
            # asset_id = asset_id[0] #! use the norm stats from the first episode
            for a_id in asset_id:
                data_assets_dir = str(assets_dir / a_id)
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
                all_norm_stats.append(norm_stats)
            
            agg = {}
            
            for key in all_norm_stats[0].keys():
                from openpi.shared.normalize import NormStats
                agg[key] = NormStats(
                    mean = np.mean([norm_stats[key].mean for norm_stats in all_norm_stats], axis=0), 
                    std = np.mean([norm_stats[key].std for norm_stats in all_norm_stats], axis=0),
                    q01 = np.mean([norm_stats[key].q01 for norm_stats in all_norm_stats], axis=0),
                    q99 = np.mean([norm_stats[key].q99 for norm_stats in all_norm_stats], axis=0),
                )


            norm_stats = agg

            # data_assets_dir = str(assets_dir / asset_id)
            # norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            # logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotVLABenchDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[vlabench_policy.VLABenchInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[vlabench_policy.VLABenchOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotACOTVLABenchDataConfig(DataConfigFactory):
    repo_id: Union[str, Sequence[str]] = "..."
    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[vlabench_policy.VLABenchACOTInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
                ],
            outputs=[vlabench_policy.VLABenchACOTOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        ret_config =  dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotACOTGo1DataConfig(DataConfigFactory):
    """
    Configuration for the Go1 robot dataset.
    This config handles the data transforms for the Go1 robot's multi-camera setup and state/action space.
    """
    prompt_from_hl_instruction: bool = False

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action"
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: list(_transforms.make_bool_mask(-14, 18))
    )
    action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: list(_transforms.make_bool_mask(-16, 16))
    )
    delta_action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: _transforms.make_bool_mask(14, -18)
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[go1_policy.Go1ACOTInputs(
                action_dim=model_config.action_dim,
                state_mask = self.state_mask,
                action_mask = self.action_mask,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
            ],
            outputs=[go1_policy.Go1ACOTOutputs()],
        )

        # Apply delta action transform if enabled
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(self.delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(self.delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        ret_config =  dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotACOTGo2DataConfig(DataConfigFactory):
    """
    Configuration for the Go2 robot dataset.
    This config handles the data transforms for the Go2 robot's multi-camera setup and state/action space.
    """
    prompt_from_hl_instruction: bool = False

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action"
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask: Sequence[int] = dataclasses.field(
        # mask two gripper state and useless four waist joint states.
        default_factory=lambda: _transforms.make_bool_mask(-14, 2, 4, -1, 11)
    )
    action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: _transforms.make_bool_mask(-16, 4, -1, 11)
    )
    delta_action_mask: Sequence[int] = dataclasses.field(
        default_factory=lambda: _transforms.make_bool_mask(14, -18)
    )
    prompt_map_inject_to_training: Dict[str, Sequence[str]] = dataclasses.field(
        default_factory=lambda: {}
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[go2_policy.Go2ACOTInputs(
                action_dim=model_config.action_dim,
                state_mask = self.state_mask,
                action_mask = self.action_mask,
                prompt_map_inject_to_training = self.prompt_map_inject_to_training,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
            ],
            outputs=[go2_policy.Go2ACOTOutputs()],
        )

        # Apply delta action transform if enabled
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(self.delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(self.delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        ret_config =  dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config

@dataclasses.dataclass(frozen=True)
class LeRobotACOTLiberoDataConfig(DataConfigFactory):

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoACOTInputs(
                model_type=model_config.model_type,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
                ],
            outputs=[libero_policy.LiberoACOTOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        model_transforms = ModelTransformFactory()(model_config)
        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LeRobotACOTLiberoPlusDataConfig(DataConfigFactory):

    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoACOTInputs(
                model_type=model_config.model_type,
                acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts))
                ],
            outputs=[libero_policy.LiberoACOTOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        model_transforms = ModelTransformFactory()(model_config)
        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LerobotAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # mask state out (set to all zeros)
    mask_state: bool = False

    # if convert to eef position
    convert_to_eef_position: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                    convert_to_eef_position=self.convert_to_eef_position,
                )
            ],
            outputs=[agilex_policy.AgilexOutputs()],
        )

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            # Assuming first 13 dimensions are joints and last dimension is gripper
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class LerobotACOTAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # if convert to eef position
    convert_to_eef_position: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexACOTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    convert_to_eef_position=self.convert_to_eef_position,
                    acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts)
                )
            ],
            outputs=[agilex_policy.AgilexACOTOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class LerobotARXDataConfig(DataConfigFactory):
    """
    Configuration for the Lerobot ARX dataset.
    This config handles the data transforms for the Lerobot ARX's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask = np.array(_transforms.make_bool_mask(-14, 18))
    action_mask = np.array(_transforms.make_bool_mask(-14, 18))

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ARXInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mask=self.state_mask,
                    action_mask=self.action_mask,
                )
            ],
            outputs=[arx_policy.ARXOutputs()],
        )

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)

        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class LerobotACOTARXDataConfig(DataConfigFactory):
    """
    Configuration for the Lerobot ARX dataset.
    This config handles the data transforms for the Lerobot ARX's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    extra_delta_transform: Sequence[bool] = (False, False)
    joint_action_shifts: Sequence[int] = (2, 1)

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    state_mask = np.array(_transforms.make_bool_mask(-14, 18))
    action_mask = np.array(_transforms.make_bool_mask(-14, 18))

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ARXACOTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mask=self.state_mask,
                    action_mask=self.action_mask,
                    acot_action_generation=((model_config.coarse_action_horizon, model_config.action_horizon), self.joint_action_shifts)
                )
            ],
            outputs=[arx_policy.ARXACOTOutputs()],
        )

        # Apply delta action transform if enabled
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.ACOTDeltaActions(delta_action_mask, self.extra_delta_transform)],
            outputs=[_transforms.ACOTAbsoluteActions(delta_action_mask, self.extra_delta_transform)],
        )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        ret_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
        object.__setattr__(ret_config, 'joint_action_shifts', self.joint_action_shifts)
        return ret_config


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "ACOT-VLA"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        if not os.getenv("DEBUG_MODE", default=False) == "true":
            return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
        else:
            return (pathlib.Path(self.checkpoint_base_dir) / self.name / "debug").resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="teamace_mora",
        project_name="ACOT-MultiLoRA",
        model=acot_multilora.ACOTMultiLoRAConfig(
            coarse_action_horizon=60,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            adopt_explicit_action_reasoner=True,
            adopt_implicit_action_reasoner=True,
            downsample_based_implicit_extractor=True,
            # --- Multi-LoRA knobs ---
            num_loras=4,               
            lora_rank=32,
            lora_alpha=None,
            forced_cluster_id=None,
            use_shared_lora=True,
            router_hidden_dim=256,
            router_temperature=1.0,
            router_balance_weight=0.002,   
            router_z_loss_weight=1e-3,
        ),
        data=LerobotACOTGo2DataConfig(
            default_prompt="",
            repo_id=[
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_addition/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_part_1/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/hold_pot/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/open_door/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/place_block_into_box",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/pour_workpiece/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_1/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_3/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/stock_and_straighten_shelf",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/stock_and_straighten_shelf_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/take_wrong_item_shelf",
            ],
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/workspace/acot_vla/norm_stats/",
            ),
            # prompt_map_inject_to_training = {
            #     # task name: (prompt to replace vanilla annotation, probability to replace)
            #     "Unload workpiece_icra_SIM": ("Pour the workpiece into the box", 0.5),
            #     "Turn the doorknob": ("Turn the doorknob and push the door", 0.5),
            #     "Make popcorn": ("Scoop the popcorn and pour it into the popcorn bucket", 0.5),
            #     "Carry the pot": ("Grasp the two handles of the pot and place it on the stove", 0.5),

            #     "Insert building block holes_2_SIM": (
            #         "Pick up the yellow circular block from the table, "
            #         "and place it into the round hole of the block box",
            #         0.2
            #     ),
            #     "Remove misplaced beverages from shelves": (
            #         "Pick up the incorrectly placed item from the shelf, "
            #         "and place it into the shopping basket",
            #         0.2
            #     ),
            #     "Stock supermarket shelves  \nStraighten products  \nAttend ICRA conference  \nOperate SIM card": (
            #         "Pick up the wei-chuan orange juice in the shopping basket, "
            #         "and place it on the shelf. "
            #         "Then, straighten the toppled wei-chuan grape juice",
            #         0.2
            #     ),
            #     "Sort packages": (
            #         "Grab the <color> package on the table, "
            #         "turn the waist right to face the barcode scanner, "
            #         "place the package on the scanning table with the barcode facing up. "
            #         "Then, grab the package, "
            #         "rotate the waist and place the package in the blue bin. "
            #         "Finally, return the waist back to face the initial table",
            #         0.2
            #     ),

            #     "Clear the desktop": (
            #         "Pick up the pen on the left side and place it into the pen holder, "
            #         "close the laptop, "
            #         "pick up the tissue on the table and place it into the trash bin on the right size. "
            #         "Then, pick up the mouse and place it on the right side of the laptop. "
            #         "Finally, straighten the colored pencil box",
            #         0.5
            #     ),
            # },
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top_head":   "observation.images.top_head",
                                "hand_left":  "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "task": "task",
                            "episode_index": "episode_index",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                dataloader_sampler="",
                prompt_from_hl_instruction=False,
            ),
            joint_action_shifts=(2, 1),
            extra_delta_transform=(True, True),
            delta_action_mask=_transforms.make_bool_mask(14, -18),
        ),
        # --- LR schedule: short warmup, decay across the whole run ---
        # warmup_steps + decay_steps == num_train_steps, so LR is monotonically
        # decreasing after step 300 until the very last step.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            "/workspace/acot_vla/checkpoints/ace_v1/12000/params"
        ),
        num_train_steps=20_000,
        save_interval=2_000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 200,
        num_workers=16,             
        batch_size=24,              
        log_interval=50,
        freeze_filter=acot_multilora.ACOTMultiLoRAConfig(
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(
            freeze_vision=True,
            freeze_llm=True,
            freeze_llm_embedder=True,
            freeze_dual_ae=(True, True),           
            freeze_action_head_base=True,          # new knob, default True
        ),
    ),
    TrainConfig(
        name="teamace_multilora",
        # For the ICRA sim challenge, we set both coarse and fine action horizons to 30 since the tasks are relatively long-horizon.
        # We also use both explicit and implicit action reasoners, and use the downsample-based implicit extractor.
        # You can modify these design choices based on the specific tasks and dataset. 
        model=acot_vla.ACOTConfig(coarse_action_horizon=30, action_horizon=30, paligemma_variant="gemma_2b_lora", adopt_explicit_action_reasoner=True, adopt_implicit_action_reasoner=True, downsample_based_implicit_extractor=True),
        data=LerobotACOTGo2DataConfig(
            default_prompt = "This is the icra simulation challenge baseline config.",
            # Fill in the 9 tasks for training. You can use all 9 tasks, or a subset of them based on your preference.
            repo_id = [
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_addition/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_part_1/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/hold_pot/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/open_door/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/place_block_into_box",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/pour_workpiece/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_1/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_3/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/stock_and_straighten_shelf",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/stock_and_straighten_shelf_part_2/",
                "/workspace/acot_vla/AgiBotWorld/Reasoning2Action-Sim/dataset_without_depth/take_wrong_item_shelf",
            ],
            # Set the asset dir to specify normalization stats calculated from the dataset
            assets=AssetsConfig(
                assets_dir=None,
                asset_id="/workspace/acot_vla/norm_stats/",
            ),
            # this line defines a mapping from task name to (prompt, probability of replacement) for training. 
            # If the current episode's task name matches one of the keys in the mapping, then with the corresponding probability, 
            # the original prompt (loaded from the dataset based on the task name) will be replaced with the provided prompt. 
            # This allows for more diverse and potentially more informative prompts during training.
            prompt_map_inject_to_training = {
                # task name: (prompt to replace vanilla annotation, probability to replace)
                "Unload workpiece_icra_SIM": ("Pour the workpiece into the box", 0.5),
                "Turn the doorknob": ("Turn the doorknob and push the door", 0.5),
                "Make popcorn": ("Scoop the popcorn and pour it into the popcorn bucket", 0.5),
                "Carry the pot": ("Grasp the two handles of the pot and place it on the stove", 0.5),

                "Insert building block holes_2_SIM": (
                    "Pick up the yellow circular block from the table, "
                    "and place it into the round hole of the block box",
                    0.2
                ),
                "Remove misplaced beverages from shelves": (
                    "Pick up the incorrectly placed item from the shelf, "
                    "and place it into the shopping basket",
                    0.2
                ),
                "Stock supermarket shelves  \nStraighten products  \nAttend ICRA conference  \nOperate SIM card": (
                    "Pick up the wei-chuan orange juice in the shopping basket, "
                    "and place it on the shelf. "
                    "Then, straighten the toppled wei-chuan grape juice",
                    0.2
                ),
                "Sort packages": (
                    "Grab the <color> package on the table, "
                    "turn the waist right to face the barcode scanner, "
                    "place the package on the scanning table with the barcode facing up. "
                    "Then, grab the package, "
                    "rotate the waist and place the package in the blue bin. "
                    "Finally, return the waist back to face the initial table",
                    0.2
                ),

                "Clear the desktop": (
                    "Pick up the pen on the left side and place it into the pen holder, "
                    "close the laptop, "
                    "pick up the tissue on the table and place it into the trash bin on the right size. "
                    "Then, pick up the mouse and place it on the right side of the laptop. "
                    "Finally, straighten the colored pencil box",
                    0.5
                ),
            },
            repack_transforms =_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top_head": "observation.images.top_head",
                                "hand_left": "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            # repack task name and episode id here for specific prompt replacement in training
                            "task": "task",
                            "episode_index": "episode_index"
                        }
                    )
                ]
            ),
            # this line allows using episode level annotation for training, essential for instruction following
            # base_config = DataConfig(dataloader_sampler = "subtask", prompt_from_hl_instruction = True),
            base_config = DataConfig(prompt_from_task=True, dataloader_sampler = "", prompt_from_hl_instruction = False),
            # this line is important for action cot training, it shifts the action sequence by a certain number of steps 
            # to create the input for the coarse action reasoner and the final action head. 
            # You can tune these values based on the characteristics of your dataset. 
            # Generally, we find that having a small positive shift (e.g. 1 or 2) works well.
            joint_action_shifts = (2, 1),
            # if using delta action to train model is excepted, set True in extra_delta_transform
            extra_delta_transform = (True, True),
            # delta_action_mask controls which dimensions of the action are used for delta action transformation.
            delta_action_mask = _transforms.make_bool_mask(14, -18)
        ),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 1e-4,
            decay_steps = 4_000,
            decay_lr = 5e-5,
        ),
        optimizer = _optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay = 0.999,
        weight_loader = weight_loaders.ACOTCheckpointWeightLoader(
            "/workspace/acot_vla/checkpoints/ace_v1/12000/params"
        ),
        num_train_steps = 5_000,
        save_interval = 1000 if not os.getenv("DEBUG_MODE", default=False) == "true" else 200,
        num_workers = 8,
        batch_size = 64,
        # You can select to freeze certain parts of the model during training by setting the corresponding flags to True
        freeze_filter = acot_vla.ACOTConfig(paligemma_variant="gemma_2b_lora").get_freeze_filter(
            freeze_vision = True, freeze_llm = True, freeze_llm_embedder=True, freeze_dual_ae=[False, False]
        )
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
