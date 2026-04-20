import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable
import jax
import flax.traverse_util
import numpy as np
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")

@dataclasses.dataclass(frozen=True)
class ACOTCheckpointWeightLoader(WeightLoader):
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        key_mapping = {
            "action_in_proj": "coarse_action_in_proj",
            "action_out_proj": "coarse_action_out_proj",

            "time_mlp_in": "coarse_time_mlp_in",
            "time_mlp_out": "coarse_time_mlp_out",

            "action_time_mlp_in": "coarse_action_time_mlp_in",
            "action_time_mlp_out": "coarse_action_time_mlp_out",
        }

        keys_to_check = list(key_mapping.keys())
        # Specially, we use pre-trained weight to initialize coarse&explicit action reasoner to stablize training
        for source_key in keys_to_check:
            if source_key in loaded_params:
                target_key = key_mapping[source_key]
                loaded_params[target_key] = loaded_params[source_key]
                print(f"[INFO] Re-mapped pretrained weight '{source_key}' -> '{target_key}' (for Reasoner)")

        # --- BEGIN FIX: normalize list-index keys to integers ---------------
        # Orbax saves Python lists as dicts with stringified integer keys
        # ("0", "1", ...), while nnx.state() flattens Python lists using actual
        # integer indices. That mismatch makes `flat_loaded` keys like
        #   ('implicit_action_reasoner', 'q_proj', '0', 'kernel')
        # never match `flat_ref` keys like
        #   ('implicit_action_reasoner', 'q_proj',  0 , 'kernel').
        # Result: every list-indexed submodule silently loads as random.
        # We recursively rewrite stringified-int dict keys back to ints before
        # handoff to _merge_params so the tuple-key comparison succeeds.
        def _normalize_list_indices(obj):
            if isinstance(obj, dict):
                # If ALL keys in this dict look like non-negative integer
                # strings, treat it as a serialized list and convert keys
                # to ints. Otherwise leave keys as-is.
                keys = list(obj.keys())
                all_intlike = len(keys) > 0 and all(
                    isinstance(k, str) and k.isdigit() for k in keys
                )
                out = {}
                for k, v in obj.items():
                    new_k = int(k) if all_intlike else k
                    out[new_k] = _normalize_list_indices(v)
                return out
            return obj

        loaded_params = _normalize_list_indices(loaded_params)
        # --- END FIX --------------------------------------------------------

        return _merge_params(loaded_params, params, missing_regex=".*")

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _align_param(expected, loaded, init_method):
    if expected.shape == loaded.shape:
        return loaded.astype(expected.dtype)

    min_shape = tuple(min(e, l) for e, l in zip(expected.shape, loaded.shape))
    slices = tuple(slice(0, m) for m in min_shape)

    if init_method == "zeros":
        new_param = jnp.zeros(expected.shape, dtype=expected.dtype)
    elif init_method == "random":
        new_param = jax.random.normal(jax.random.PRNGKey(0), expected.shape, dtype=expected.dtype) * 0.02
    else:
        raise ValueError(f"Unknown init method: {init_method}")

    new_param = new_param.at[slices].set(loaded[slices])
    print(f"[WARN] Shape mismatch: expected {expected.shape}, got {loaded.shape}, truncated to {min_shape}")
    return new_param

def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str, init="random") -> at.Params:

    flat_ref = flax.traverse_util.flatten_dict(params, sep=None)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep=None)

    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = _align_param(flat_ref[k], v, init)

    pattern = re.compile(missing_regex)

    missing_keys = {k for k in flat_ref if pattern.fullmatch("/".join(map(str, k))) and k not in result}
    
    for k in missing_keys:
        key_path = "/".join(map(str, k))
        expected_param = flat_ref[k]

        cloned = False
        cloned_path_source = re.sub(r'(\w+)\_(\d+)', r'\g<1>_1', key_path, count=1)
        k_source = tuple(cloned_path_source.split('/'))

        if cloned_path_source != key_path:
            if k_source in flat_loaded:
                loaded_param_source = flat_loaded[k_source]

                if expected_param.shape == loaded_param_source.shape:
                    result[k] = loaded_param_source.astype(expected_param.dtype)
                    print(f"[INFO] Cloned missing param {key_path} from {cloned_path_source} {expected_param.shape}")
                    cloned = True
                else:
                    print(f"[WARN] Clone attempt failed for {key_path}: source shape {loaded_param_source.shape} != target shape {expected_param.shape}")

        if not cloned:
            if init == "zeros":
                result[k] = jnp.zeros(expected_param.shape, dtype=expected_param.dtype)
            else:
                result[k] = jax.random.normal(jax.random.PRNGKey(0), expected_param.shape, dtype=expected_param.dtype) * 0.02
            print(f"[WARN] Missing param {key_path}, init as {init}, {expected_param.shape}")

    return flax.traverse_util.unflatten_dict(result)