import dataclasses
import functools
import logging
import platform
import signal
import sys
from typing import Any
import os
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


# =============================================================================
# Parameter summary (PyTorch/torchinfo-style table)
# =============================================================================
def _path_to_str(path_tuple) -> str:
    """Join a pytree path tuple into a displayable string. Works with ints
    (scanned/stacked layers produce integer keys like ('layers', 0, 'kernel')),
    jax tree DictKey/SequenceKey/GetAttrKey objects, and plain strings."""
    parts = []
    for p in path_tuple:
        # jax.tree_util key types
        if hasattr(p, "key"):           # DictKey
            parts.append(str(p.key))
        elif hasattr(p, "idx"):         # SequenceKey
            parts.append(str(p.idx))
        elif hasattr(p, "name"):        # GetAttrKey
            parts.append(str(p.name))
        else:
            parts.append(str(p))
    return "/".join(parts)


def _iter_param_rows(model, trainable_filter):
    """Yield (path, shape, dtype, n_params, is_trainable) for every nnx.Param in model."""
    all_state = nnx.state(model, nnx.Param)
    trainable_state = nnx.state(model, trainable_filter)

    all_pure = all_state.to_pure_dict()
    trainable_pure = trainable_state.to_pure_dict()

    # Don't pass sep="/" — some path components are ints (scanned layers),
    # and flax's flatten_dict will try to str-join them and crash. Flatten to
    # tuple keys and stringify ourselves.
    all_flat = traverse_util.flatten_dict(all_pure)
    trainable_flat = traverse_util.flatten_dict(trainable_pure)
    trainable_paths = {_path_to_str(k) for k in trainable_flat.keys()}

    for path_tuple, arr in all_flat.items():
        path = _path_to_str(path_tuple)
        shape = tuple(arr.shape)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        dtype = str(arr.dtype)
        is_trainable = path in trainable_paths
        yield path, shape, dtype, n, is_trainable


def _dtype_bytes(dtype_str: str) -> int:
    try:
        return jnp.dtype(dtype_str).itemsize
    except Exception:
        return 4  # fallback


def print_parameter_summary(model, trainable_filter, logger=None):
    """Print a PyTorch-style parameter summary table with trainable/frozen breakdown."""
    log = logger.info if logger is not None else logging.info

    rows = list(_iter_param_rows(model, trainable_filter))

    total_params = 0
    trainable_params = 0
    total_bytes = 0
    trainable_bytes = 0

    for _, _, dtype, n, tr in rows:
        nb = n * _dtype_bytes(dtype)
        total_params += n
        total_bytes += nb
        if tr:
            trainable_params += n
            trainable_bytes += nb
    frozen_params = total_params - trainable_params
    frozen_bytes = total_bytes - trainable_bytes

    # Dynamic column widths, but keep the table readable.
    name_w = min(max((len(r[0]) for r in rows), default=20), 90)
    name_w = max(name_w, 40)
    shape_w = max((len(str(r[1])) for r in rows), default=10)
    shape_w = max(shape_w, 15)
    dtype_w = max((len(r[2]) for r in rows), default=7)
    dtype_w = max(dtype_w, 8)

    header = (
        f"{'Layer (path)':<{name_w}}  "
        f"{'Shape':<{shape_w}}  "
        f"{'Dtype':<{dtype_w}}  "
        f"{'# Params':>15}  "
        f"{'Trainable':>10}"
    )
    bar = "=" * len(header)
    dash = "-" * len(header)

    log(bar)
    log("Parameter Summary".center(len(bar)))
    log(bar)
    log(header)
    log(dash)
    for path, shape, dtype, n, tr in rows:
        display_path = path if len(path) <= name_w else ("..." + path[-(name_w - 3):])
        log(
            f"{display_path:<{name_w}}  "
            f"{str(shape):<{shape_w}}  "
            f"{dtype:<{dtype_w}}  "
            f"{n:>15,}  "
            f"{('Yes' if tr else 'No'):>10}"
        )
    log(dash)

    def _mb(b):
        return b / (1024 ** 2)

    if total_params > 0:
        log(f"Total params:     {total_params:>15,}  ({_mb(total_bytes):>10.2f} MB)")
        log(
            f"Trainable params: {trainable_params:>15,}  "
            f"({_mb(trainable_bytes):>10.2f} MB)  ({100 * trainable_params / total_params:5.2f}%)"
        )
        log(
            f"Frozen params:    {frozen_params:>15,}  "
            f"({_mb(frozen_bytes):>10.2f} MB)  ({100 * frozen_params / total_params:5.2f}%)"
        )
    else:
        log("No nnx.Param variables found in model.")
    log(bar)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_bytes": total_bytes,
        "trainable_bytes": trainable_bytes,
        "frozen_bytes": frozen_bytes,
    }


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

@at.typecheck
def acot_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions,
        coarse_actions: _model.CoarseActions
    ):
        out = model.compute_loss(rng, observation, actions, coarse_actions, train=True)
        # Multi-LoRA returns (loss, stats_dict); plain ACOT returns just a
        # scalar. Normalise to (loss, dict) so has_aux=True works for both.
        if isinstance(out, tuple):
            return out
        return out, {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, coarse_actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **aux,  # router/* diagnostics + loss/flow + loss/aux (empty for plain ACOT)
    }
    return new_state, info


# =============================================================================
# Graceful-shutdown helpers
# =============================================================================
class _StopFlag:
    """Tiny mutable flag for the SIGINT handler to toggle. Using an object (not
    a bare global) keeps the intent obvious and avoids `global` statements."""
    def __init__(self):
        self.requested = False
        self.hits = 0


def _install_sigint_handler(stop_flag: _StopFlag):
    """Install a SIGINT (Ctrl+C) handler that requests a graceful stop.

    Why a handler instead of `try/except KeyboardInterrupt`:
      - JAX dispatch is async. An exception fired mid-dispatch can leave the
        train_state references in an inconsistent / half-donated state.
      - The data loader has worker processes; an exception from inside
        `next(data_iter)` can be noisy and leaves checkpoint saving to luck.
      - With a flag, we always finish the current step, THEN save cleanly.

    Second Ctrl+C -> hard exit (escape hatch if saving itself hangs).
    """
    def handler(signum, frame):
        stop_flag.hits += 1
        if stop_flag.hits >= 2:
            logging.error("Second interrupt received — force exiting without saving.")
            # 128 + SIGINT(2) is the conventional exit code.
            os._exit(130)
        stop_flag.requested = True
        logging.warning(
            "Interrupt received — will save an emergency checkpoint after the "
            "current step finishes. Press Ctrl+C again to force-exit."
        )

    signal.signal(signal.SIGINT, handler)


def _emergency_save(checkpoint_manager, train_state, data_loader, step: int):
    """Block on pending JAX work, then persist state + wait for orbax to flush."""
    logging.warning("=" * 70)
    logging.warning(f"Saving emergency checkpoint at step {step} ...")
    logging.warning("=" * 70)
    try:
        # Make sure the train_state arrays are fully materialized before orbax
        # tries to serialize them — otherwise async dispatch can race the save.
        jax.block_until_ready(train_state)
        _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
        checkpoint_manager.wait_until_finished()
        logging.info(f"Emergency checkpoint saved successfully at step {step}.")
        return True
    except Exception as e:
        # Orbax may refuse if `step` collides with an in-flight async save or
        # violates its save policy. Log loudly so the user knows.
        logging.error(f"Emergency checkpoint save FAILED: {e!r}")
        return False


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
        ptrain_step = jax.jit(
            functools.partial(acot_train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)

    # ------------------------------------------------------------------
    # PyTorch-style parameter summary (trainable + frozen breakdown).
    # ------------------------------------------------------------------
    model_for_summary = nnx.merge(train_state.model_def, train_state.params)
    summary_stats = print_parameter_summary(model_for_summary, config.trainable_filter)
    # Also push the headline numbers to wandb so they show up in the run overview.
    try:
        wandb.summary["params/total"] = summary_stats["total_params"]
        wandb.summary["params/trainable"] = summary_stats["trainable_params"]
        wandb.summary["params/frozen"] = summary_stats["frozen_params"]
        wandb.summary["params/total_MB"] = summary_stats["total_bytes"] / (1024 ** 2)
        wandb.summary["params/trainable_MB"] = summary_stats["trainable_bytes"] / (1024 ** 2)
    except Exception as e:
        logging.warning(f"Could not write param summary to wandb: {e!r}")
    # Free the temporary merged model so we don't keep extra references alive.
    del model_for_summary

    # ------------------------------------------------------------------
    # Install SIGINT (Ctrl+C) handler for graceful emergency-save.
    # ------------------------------------------------------------------
    stop_flag = _StopFlag()
    _install_sigint_handler(stop_flag)

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    last_completed_step = start_step
    try:
        for step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)
            last_completed_step = int(train_state.step)

            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                infos = []
            batch = next(data_iter)

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

            # Graceful shutdown check — runs AFTER the step fully completes.
            if stop_flag.requested:
                pbar.close()
                saved = _emergency_save(
                    checkpoint_manager, train_state, data_loader, last_completed_step
                )
                logging.info(
                    f"Exiting after Ctrl+C. Last completed step: {last_completed_step}. "
                    f"Emergency save: {'OK' if saved else 'FAILED'}."
                )
                # Clean up data loader workers if the loader exposes a shutdown hook.
                _shutdown_data_loader(data_loader)
                # Exit with conventional SIGINT code.
                sys.exit(0 if saved else 130)

    except KeyboardInterrupt:
        # Belt-and-suspenders: if the signal handler somehow didn't catch it
        # (e.g., raised between the flag check and the next step), still save.
        pbar.close()
        logging.warning("KeyboardInterrupt reached the training loop — saving checkpoint.")
        saved = _emergency_save(
            checkpoint_manager, train_state, data_loader, last_completed_step
        )
        _shutdown_data_loader(data_loader)
        sys.exit(0 if saved else 130)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    _shutdown_data_loader(data_loader)


def _shutdown_data_loader(data_loader):
    """Best-effort cleanup of data-loader worker processes.

    Different data-loader backends (PyTorch, grain, custom) expose different
    shutdown hooks. We probe for the common ones and stay silent if none match.
    """
    for attr in ("shutdown", "close", "_shutdown_workers"):
        fn = getattr(data_loader, attr, None)
        if callable(fn):
            try:
                fn()
                logging.info(f"Data loader {attr}() called successfully.")
                return
            except Exception as e:
                logging.warning(f"Data loader {attr}() raised {e!r}; continuing.")


if __name__ == "__main__":
    main(_config.cli())