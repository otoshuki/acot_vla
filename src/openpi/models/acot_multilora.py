"""ACoT-VLA with Multi-LoRA specialization for task groups.

This model extends the standard ACOT_VLA by attaching N parallel LoRA adapters
to the Explicit Action Reasoner (EAR), Implicit Action Reasoner (IAR), and the
action-projection layers. The language backbone (PaliGemma LLM) and the vision
encoder (SigLIP) are NEVER touched by these extra LoRAs.

At forward time, a one-hot "gate" of shape (B, N) selects exactly one LoRA per
sample. The gate is produced by a small LEARNED router MLP that reads the
mean-pooled Gemma input-token embedding and outputs a softmax over N experts.
Top-1 is taken via a straight-through estimator so gradients flow into the
router end-to-end. During training we add Gumbel noise for exploration and
two MoE-style auxiliary losses (Switch-Transformer load-balancing + router
z-loss) to prevent expert collapse and keep logits numerically stable.
`centroids_path` is retained as an unused field for backward-compat with the
old config; the precomputed-centroid path has been retired.

Design notes:
  * Only pi05 + downsample_based_implicit_extractor is supported — matches the
    user's `teamace_multilora` config. Adding other variants is a small copy.
  * `ACOTMultiLoRAConfig` subclasses `ACOTConfig` so existing openpi data
    pipelines (ModelTransformFactory, DataConfigFactory, etc.) work unchanged.
  * `model_type` is the same as the parent (ACOT_VLA_PI05), so train.py's
    existing `acot_train_step` path is used — no training code changes.
  * Because the gate is one-hot, the effective per-sample LoRA weight is a
    single lookup; compute cost of multi-LoRA ≈ single LoRA.
  * LoRA B is initialised to 0, so at step 0 the model is exactly the base
    (pretrained) model.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.pi0 import posemb_sincos, make_attn_mask
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.models.acot_vla import ACOTConfig

logger = logging.getLogger("ACoT_MultiLoRA")


# ---------------------------------------------------------------------------
# Multi-LoRA primitives
# ---------------------------------------------------------------------------

class MultiLoRALinear(nnx.Module):
    """Dense layer with N specialist LoRA adapters + optional always-on shared adapter.

    Layout:
      * `kernel`, `bias` — plain Linear params, at top level (match nnx.Linear
        storage so a pretrained checkpoint loads cleanly).
      * `lora_A` (N, r, in), `lora_B` (N, out, r) — task specialists, selected
        by a one-hot `gate` of shape (B, N).
      * If `use_shared_lora=True`: additional `lora_A_shared` (r, in) and
        `lora_B_shared` (out, r) that are applied on EVERY forward, no gate.
        Useful for generic task-agnostic adaptation that complements the
        per-group specialists.

    Forward (conceptual):
        delta_specialist[b] = B[c(b)] @ A[c(b)] @ x[b]     # one-hot routed
        delta_shared[b]     = B_shared  @ A_shared  @ x[b] # always applied
        y = W x + bias + scaling * (delta_specialist + delta_shared)

    Both specialist and shared B are zero-initialised, so at step 0 the delta
    is exactly zero ⇒ the model starts as the pretrained base.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_loras: int,
        lora_rank: int,
        *,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        lora_alpha: float | None = None,
        use_shared_lora: bool = False,
    ):
        self.num_loras = num_loras
        self.lora_rank = lora_rank
        self.in_features = in_features
        self.out_features = out_features
        self.use_shared_lora = use_shared_lora
        alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self.scaling = alpha / float(lora_rank)

        # --- Base Linear params stored at top level to match nnx.Linear ---
        std_w = 1.0 / (in_features ** 0.5)
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features), dtype=param_dtype) * std_w
        )
        self.bias = nnx.Param(jnp.zeros((out_features,), dtype=param_dtype))

        # --- Per-cluster specialist LoRA adapters ---
        std_a = 1.0 / (in_features ** 0.5)
        self.lora_A = nnx.Param(
            jax.random.normal(rngs.params(), (num_loras, lora_rank, in_features), dtype=param_dtype) * std_a
        )
        self.lora_B = nnx.Param(
            jnp.zeros((num_loras, out_features, lora_rank), dtype=param_dtype)
        )

        # --- Optional always-on shared LoRA adapter ---
        if use_shared_lora:
            self.lora_A_shared = nnx.Param(
                jax.random.normal(rngs.params(), (lora_rank, in_features), dtype=param_dtype) * std_a
            )
            self.lora_B_shared = nnx.Param(
                jnp.zeros((out_features, lora_rank), dtype=param_dtype)
            )

    def __call__(self, x: jax.Array, gate: Optional[jax.Array] = None) -> jax.Array:
        # Plain linear: y = x @ kernel + bias  (matches nnx.Linear exactly).
        kernel = self.kernel.value
        bias = self.bias.value
        base_out = jnp.matmul(x, kernel.astype(x.dtype)) + bias.astype(x.dtype)

        if gate is None:
            # No gate provided → skip BOTH specialist and shared LoRAs. Used
            # for "raw pretrained behaviour" smoke checks.
            return base_out

        B_size = gate.shape[0]
        orig_shape = x.shape
        x_flat = x.reshape(B_size, -1, orig_shape[-1])                         # (B, T, in)

        # --- Specialist delta (per-sample, one-hot routed) ---
        A_eff = jnp.einsum("bn,nri->bri", gate.astype(self.lora_A.value.dtype), self.lora_A.value)
        B_eff = jnp.einsum("bn,nor->bor", gate.astype(self.lora_B.value.dtype), self.lora_B.value)
        Ax = jnp.einsum("bri,bti->btr", A_eff.astype(x_flat.dtype), x_flat)    # (B, T, r)
        delta_flat = jnp.einsum("bor,btr->bto", B_eff.astype(Ax.dtype), Ax)    # (B, T, out)

        # --- Shared delta (always on, no gate) ---
        if self.use_shared_lora:
            A_s = self.lora_A_shared.value.astype(x_flat.dtype)                # (r, in)
            B_s = self.lora_B_shared.value.astype(x_flat.dtype)                # (out, r)
            Ax_s = jnp.einsum("ri,bti->btr", A_s, x_flat)                      # (B, T, r)
            delta_flat = delta_flat + jnp.einsum("or,btr->bto", B_s, Ax_s)     # (B, T, out)

        delta = delta_flat.reshape(orig_shape[:-1] + (self.out_features,))
        return base_out + (self.scaling * delta).astype(base_out.dtype)


class MultiLoRAMLP(nnx.Module):
    """Matches acot_vla.MLP interface but each fc is a MultiLoRALinear."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_loras: int,
        lora_rank: int,
        *,
        activate: bool = True,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        use_shared_lora: bool = False,
    ):
        kw = dict(num_loras=num_loras, lora_rank=lora_rank, rngs=rngs,
                  param_dtype=param_dtype, use_shared_lora=use_shared_lora)
        self.fc1 = MultiLoRALinear(input_dim, hidden_dim, **kw)
        self.fc2 = MultiLoRALinear(hidden_dim, hidden_dim, **kw)
        self.fc3 = MultiLoRALinear(hidden_dim, output_dim, **kw)
        self.activate = activate

    def __call__(self, x: jax.Array, gate: Optional[jax.Array] = None) -> jax.Array:
        if self.activate:
            return self.fc3(nnx.swish(self.fc2(nnx.swish(self.fc1(x, gate)), gate)), gate)
        return self.fc3(self.fc2(self.fc1(x, gate), gate), gate)


class MultiLoRAUnifiedAttention(nnx.Module):
    """UnifiedAttentionModule with LoRA on Q/KV/Out projections.

    The internal nnx.MultiHeadAttention has no simple LoRA hook, so we leave it
    plain. Task specialisation enters via the LoRA'd projections that flank it.
    """

    def __init__(
        self,
        in_dim_1: int,
        in_dim_2: int,
        out_dim: int,
        apply_sigmoid: bool,
        num_loras: int,
        lora_rank: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        use_shared_lora: bool = False,
    ):
        kw = dict(num_loras=num_loras, lora_rank=lora_rank, rngs=rngs,
                  param_dtype=param_dtype, use_shared_lora=use_shared_lora)
        self.q_proj = MultiLoRALinear(in_dim_1, hidden_dim, **kw)
        self.kv_proj = MultiLoRALinear(in_dim_2, hidden_dim * 2, **kw)
        self.attn = nnx.MultiHeadAttention(
            in_features=hidden_dim, num_heads=num_heads, rngs=rngs, param_dtype=param_dtype
        )
        self.fc_out = MultiLoRALinear(hidden_dim, out_dim, **kw)
        self.apply_sigmoid = apply_sigmoid

    def __call__(
        self,
        feat_1: jax.Array,
        feat_2: jax.Array,
        gate: Optional[jax.Array] = None,
        decode: bool = False,
    ) -> jax.Array:
        Q = self.q_proj(feat_1, gate)
        KV = self.kv_proj(feat_2, gate)
        K, V = jnp.split(KV, 2, axis=-1)
        attn_out = self.attn(Q, K, V, decode=decode)
        out = self.fc_out(attn_out, gate)
        return nnx.sigmoid(out) if self.apply_sigmoid else out


class MultiLoRADownsampleExtractor(nnx.Module):
    """Drop-in multi-LoRA replacement for the paper's Downsample IAR.

    Matches openpi.models.acot_vla.DownsampleExtractor but every linear
    projection is a MultiLoRALinear. The group-shared layout (projections
    shared across `group_size` LLM layers) is preserved.
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        depth: int,
        num_loras: int,
        lora_rank: int,
        *,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        downsample_dim: int = 512,
        group_size: int = 3,
        num_queries: int = 1,
        heads: int = 8,
        use_shared_lora: bool = False,
    ):
        self.dim = dim
        self.depth = depth
        self.downsample_dim = downsample_dim
        self.group_size = group_size
        self.num_groups = depth // group_size
        self.num_queries = num_queries
        self.heads = heads
        self.head_dim = downsample_dim // heads

        self.query_params = [
            nnx.Param(jax.random.normal(rngs.params(), (num_queries, dim)))
            for _ in range(self.depth)
        ]
        kw = dict(num_loras=num_loras, lora_rank=lora_rank, rngs=rngs,
                  param_dtype=param_dtype, use_shared_lora=use_shared_lora)
        self.q_proj = [MultiLoRALinear(dim, downsample_dim, **kw) for _ in range(self.num_groups)]
        self.k_proj = [MultiLoRALinear(dim, downsample_dim, **kw) for _ in range(self.num_groups)]
        self.v_proj = [MultiLoRALinear(dim, downsample_dim, **kw) for _ in range(self.num_groups)]
        self.out_proj = [MultiLoRALinear(downsample_dim, output_dim, **kw) for _ in range(self.num_groups)]

    def __call__(self, K: jax.Array, V: jax.Array, gate: Optional[jax.Array] = None) -> jax.Array:
        B, L, T, D = K.shape
        outputs = []
        for l in range(L):
            g = l // self.group_size
            K_l, V_l = K[:, l, :, :], V[:, l, :, :]
            # Tile shared learnable query so it has leading batch dim for gate.
            Q_shared = self.query_params[l][None, :, :]                        # (1, Q, D)
            Q_tiled = jnp.tile(Q_shared, [B, 1, 1])                            # (B, Q, D)

            Q_proj = self.q_proj[g](Q_tiled, gate)
            Q_proj = Q_proj.reshape(B, self.num_queries, self.heads, self.head_dim).transpose(0, 2, 1, 3)

            K_proj = self.k_proj[g](K_l, gate)
            K_proj = K_proj.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

            V_proj = self.v_proj[g](V_l, gate)
            V_proj = V_proj.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

            attn = jnp.einsum("bhqd,bhkd->bhqk", Q_proj, K_proj) / jnp.sqrt(self.head_dim)
            attn = nnx.softmax(attn, axis=-1)
            pooled = jnp.einsum("bhqk,bhkd->bhqd", attn, V_proj)
            pooled = pooled.mean(axis=2) if self.num_queries > 1 else pooled.squeeze(axis=2)
            pooled = pooled.transpose(0, 2, 1).reshape(B, self.downsample_dim)
            feat = self.out_proj[g](pooled, gate)
            outputs.append(feat)

        return jnp.stack(outputs, axis=1)


# ---------------------------------------------------------------------------
# Task router
# ---------------------------------------------------------------------------

class TaskRouter(nnx.Module):
    """Small MLP router: pooled prompt embedding -> softmax over N LoRAs.

    Output-head weights are zero-initialised, so at step 0 logits are exactly
    0 and the softmax is uniform (1/N per expert) — matches the "start uniform"
    requirement. Gumbel noise (training only) provides symmetry-breaking. The
    straight-through trick inside `__call__` gives a hard one-hot gate on the
    forward pass while routing gradient through the soft probs on the backward
    pass — i.e. top-1 compute with end-to-end training.
    """

    def __init__(
        self,
        in_dim: int,
        num_loras: int,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
    ):
        self.num_loras = num_loras
        self.ln = nnx.LayerNorm(in_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        # Zero-init output head => uniform logits at step 0.
        self.fc2 = nnx.Linear(
            hidden_dim,
            num_loras,
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )

    def __call__(
        self,
        pooled: jax.Array,                      # (B, in_dim) — frozen-LLM prompt embedding
        *,
        train: bool,
        rng: Optional[jax.Array] = None,
        temperature: float = 1.0,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns (gate_ST, probs, logits).

        gate_ST  : (B, N) straight-through one-hot. Forward value is hard
                   argmax; backward gradient flows through the soft probs.
        probs    : (B, N) softmax probabilities (for aux losses + logging).
        logits   : (B, N) pre-softmax, noise-free logits (for z-loss).
        """
        h = self.ln(pooled.astype(jnp.float32))
        h = nnx.gelu(self.fc1(h))
        logits = self.fc2(h)                                         # (B, N)

        # Gumbel noise only during training — encourages early exploration.
        if train and rng is not None:
            u = jax.random.uniform(rng, logits.shape, minval=1e-9, maxval=1.0)
            gumbel = -jnp.log(-jnp.log(u))
            noised = logits + gumbel
        else:
            noised = logits

        probs = jax.nn.softmax(noised / jnp.maximum(temperature, 1e-3), axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(probs, axis=-1), self.num_loras)
        # Straight-through: forward == hard, backward == d(probs).
        gate_st = hard + probs - jax.lax.stop_gradient(probs)
        return gate_st, probs, logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ACOTMultiLoRAConfig(ACOTConfig):
    """Extends ACOTConfig with multi-LoRA + clustering settings."""

    num_loras: int = 4
    lora_rank: int = 8
    lora_alpha: float | None = None
    # DEPRECATED — retained only for backward-compat with the old config file.
    # The learned router has fully replaced the precomputed-centroid path.
    centroids_path: str | None = None
    # Debug: force a single cluster id for every sample. Overrides the router.
    forced_cluster_id: int | None = None
    # If True, each Multi-LoRA linear gains ONE extra always-on adapter (on top
    # of the `num_loras` per-cluster specialists). It receives gradient from
    # every sample regardless of gate, so it learns task-agnostic adaptation
    # while the specialists focus on per-group deltas. Total adapters per
    # linear = num_loras specialists + (1 shared if enabled).
    use_shared_lora: bool = True

    # --- Learned task router ---
    router_hidden_dim: int = 256
    router_temperature: float = 1.0
    # Switch-Transformer style load-balancing loss weight. Start small;
    # raise to ~0.05 if any expert's usage stays at 0 for >2k steps.
    router_balance_weight: float = 0.01
    # ST-MoE router z-loss (logsumexp^2) weight. Keeps router logits stable.
    router_z_loss_weight: float = 1e-3
    
    def get_freeze_filter(
        self,
        freeze_llm: bool = True,
        freeze_llm_embedder: bool = True,
        freeze_vision: bool = True,
        freeze_dual_ae=(True, True),
        freeze_action_head_base: bool = True,
    ):
        """Multi-LoRA-specific freeze filter.

        Differences from the parent:
          * Does NOT add ".*lora.*" to keep-alive, so Gemma's internal LoRA
            adapters (from paligemma_variant="gemma_2b_lora") stay FROZEN
            alongside the rest of the LLM base. Previously these were
            drifting every step and causing catastrophic forgetting.
          * Adds `freeze_action_head_base`: when True, freezes the plain
            kernel/bias/query_params of every MultiLoRALinear and every
            DownsampleExtractor/UnifiedAttention submodule. These loaded
            from the pretrained checkpoint and must not drift if we want
            LoRA to behave like LoRA.
          * Defaults are flipped to the "pure LoRA fine-tune" setting:
            everything pretrained is frozen, only the new lora_* adapters
            train. Override any argument to loosen.
        """
        # Path patterns — consistent with the parent's PathRegex usage.
        paligemma_base_filter = nnx_utils.PathRegex(r".*llm(?!.*_1|.*_2).*")
        coarse_action_expert_filter = nnx_utils.PathRegex(r".*llm.*_1.*")
        action_expert_filter = nnx_utils.PathRegex(r".*llm.*_2.*")
        img_filter = nnx_utils.PathRegex(r".*img.*")
        embedder_filter = nnx_utils.PathRegex(r".*llm.*embed.*|.*llm.*embedding.*")

        # Matches ONLY the multi-LoRA adapter params we introduced:
        # `.../lora_A`, `.../lora_B`, `.../lora_A_shared`, `.../lora_B_shared`.
        # The `$` anchor stops this from accidentally matching Gemma's own
        # internal LoRA params (which have deeper paths like `.../lora_a/kernel`).
        multilora_adapter_filter = nnx_utils.PathRegex(r".*lora_[AB](_shared)?$")

        # Learned task router — must remain trainable.
        router_filter = nnx_utils.PathRegex(r".*router.*")

        freeze_paths = []

        if freeze_vision:
            freeze_paths.append(img_filter)

        if freeze_llm:
            freeze_paths.append(paligemma_base_filter)

        if freeze_dual_ae[0]:
            freeze_paths.append(coarse_action_expert_filter)
        if freeze_dual_ae[1]:
            freeze_paths.append(action_expert_filter)

        if freeze_action_head_base:
            # "Everything that isn't the LLM, isn't the image encoder, isn't a
            # multi-LoRA adapter, and isn't the router." This catches
            # kernel/bias of every MultiLoRALinear and every Linear inside the
            # attention modules, plus query_params in the DownsampleExtractor —
            # i.e., every pretrained weight sitting outside the LLM/vision
            # tree and outside our trainable adapter/router surface.
            action_head_base_filter = nnx.All(
                nnx.Not(nnx_utils.PathRegex(r".*llm.*")),
                nnx.Not(img_filter),
                nnx.Not(multilora_adapter_filter),
                nnx.Not(router_filter),
            )
            freeze_paths.append(action_head_base_filter)

        if not freeze_paths:
            return nnx.Nothing

        base_freeze_filter = nnx.Any(*freeze_paths)

        # Keep-alive logic. We deliberately DO NOT add multilora_adapter_filter
        # here — the action_head_base_filter above already excludes it, so
        # our adapters are naturally outside freeze_paths. Adding it here
        # would redundantly un-freeze things that were never frozen.
        keep_alive_paths = []
        if freeze_llm and not freeze_llm_embedder:
            keep_alive_paths.append(embedder_filter)

        if not keep_alive_paths:
            return base_freeze_filter
        return nnx.All(base_freeze_filter, nnx.Not(nnx.Any(*keep_alive_paths)))
        
    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Same model_type as parent; train.py branches on this so we keep the
        # existing acot_train_step path.
        return super().model_type

    @override
    def create(self, rng: at.KeyArrayLike) -> "ACOTMultiLoRA_VLA":
        return ACOTMultiLoRA_VLA(self, rngs=nnx.Rngs(rng))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ACOTMultiLoRA_VLA(_model.BaseModel):
    """ACoT-VLA with N parallel LoRA adapters gated by prompt cluster.

    Public interface is identical to ACOT_VLA, so train.py's acot_train_step
    uses this unchanged.
    """

    def __init__(self, config: ACOTMultiLoRAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        assert config.pi05, "Multi-LoRA implementation is for pi05 variant only."
        self.pi05 = True
        self.num_loras = config.num_loras
        self.forced_cluster_id = config.forced_cluster_id

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        coarse_action_expert_config = _gemma.get_config(config.coarse_action_expert_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self._paligemma_width = paligemma_config.width

        # -- Backbones (frozen by filter in the user's existing config) --
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, coarse_action_expert_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # -- Action-head / projection layers (Multi-LoRA) --
        N, r = config.num_loras, config.lora_rank
        self.use_shared_lora = config.use_shared_lora
        mlk = dict(num_loras=N, lora_rank=r, rngs=rngs, use_shared_lora=config.use_shared_lora)

        self.coarse_action_in_proj = MultiLoRALinear(
            config.action_dim, coarse_action_expert_config.width, **mlk
        )
        self.action_in_proj = MultiLoRALinear(
            config.action_dim, action_expert_config.width, **mlk
        )
        self.coarse_time_mlp_in = MultiLoRALinear(
            coarse_action_expert_config.width, coarse_action_expert_config.width, **mlk
        )
        self.coarse_time_mlp_out = MultiLoRALinear(
            coarse_action_expert_config.width, coarse_action_expert_config.width, **mlk
        )
        self.time_mlp_in = MultiLoRALinear(
            action_expert_config.width, action_expert_config.width, **mlk
        )
        self.time_mlp_out = MultiLoRALinear(
            action_expert_config.width, action_expert_config.width, **mlk
        )
        self.coarse_action_out_proj = MultiLoRALinear(
            coarse_action_expert_config.width, config.action_dim, **mlk
        )
        self.action_out_proj = MultiLoRALinear(
            action_expert_config.width, config.action_dim, **mlk
        )

        # -- EAR / IAR / fusion modules (Multi-LoRA) --
        use_shared = config.use_shared_lora
        self.adopt_explicit_action_reasoner = config.adopt_explicit_action_reasoner
        if self.adopt_explicit_action_reasoner:
            self.explicit_action_reasoner = MultiLoRAUnifiedAttention(
                in_dim_1=action_expert_config.width,
                in_dim_2=coarse_action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
                num_loras=N,
                lora_rank=r,
                rngs=rngs,
                use_shared_lora=use_shared,
            )

        self.adopt_implicit_action_reasoner = config.adopt_implicit_action_reasoner
        if self.adopt_implicit_action_reasoner:
            assert config.downsample_based_implicit_extractor, (
                "Only downsample_based_implicit_extractor is supported in the multi-LoRA model."
            )
            self.implicit_action_reasoner = MultiLoRADownsampleExtractor(
                num_queries=1,
                dim=paligemma_config.head_dim,
                output_dim=action_expert_config.width,
                depth=paligemma_config.depth,
                downsample_dim=paligemma_config.head_dim // 2,
                heads=paligemma_config.num_heads,
                group_size=3,
                num_loras=N,
                lora_rank=r,
                rngs=rngs,
                use_shared_lora=use_shared,
            )
            self.implicit_action_reasoner_interact = MultiLoRAUnifiedAttention(
                in_dim_1=action_expert_config.width,
                in_dim_2=action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
                num_loras=N,
                lora_rank=r,
                rngs=rngs,
                use_shared_lora=use_shared,
            )

        if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
            self.explicit_action_reason_proj = MultiLoRALinear(
                2 * action_expert_config.width, action_expert_config.width, **mlk
            )
            self.implicit_action_reason_proj = MultiLoRALinear(
                2 * action_expert_config.width, action_expert_config.width, **mlk
            )
            self.action_reasoning_fusion = MultiLoRAUnifiedAttention(
                in_dim_1=2 * action_expert_config.width,
                in_dim_2=2 * action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
                num_loras=N,
                lora_rank=r,
                rngs=rngs,
                use_shared_lora=use_shared,
            )
        elif self.adopt_explicit_action_reasoner or self.adopt_implicit_action_reasoner:
            self.action_reasoning_fusion = MultiLoRAMLP(
                input_dim=2 * action_expert_config.width,
                hidden_dim=action_expert_config.width,
                output_dim=action_expert_config.width,
                num_loras=N,
                lora_rank=r,
                activate=False,
                rngs=rngs,
                use_shared_lora=use_shared,
            )
        else:
            self.action_reasoning_proj = None

        # -- Learned task router --
        self.router = TaskRouter(
            in_dim=paligemma_config.width,
            num_loras=config.num_loras,
            hidden_dim=config.router_hidden_dim,
            rngs=rngs,
        )
        self.router_temperature = float(config.router_temperature)
        self.router_balance_weight = float(config.router_balance_weight)
        self.router_z_loss_weight = float(config.router_z_loss_weight)

        self.deterministic = True
        self.coarse_action_horizon = config.coarse_action_horizon

    # ------------------------------------------------------------------
    # Gate computation
    # ------------------------------------------------------------------
    def _compute_gate(
        self,
        tokenized_prompt: jax.Array,
        tokenized_prompt_mask: jax.Array,
        *,
        rng: Optional[jax.Array] = None,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return (gate, probs, logits).

        gate   : (B, num_loras) straight-through one-hot routing weights.
        probs  : (B, num_loras) soft router probabilities (for aux losses).
        logits : (B, num_loras) pre-softmax logits (for z-loss).

        During inference (`train=False`, `rng=None`) the router runs
        deterministically (no Gumbel noise) and `gate` is pure argmax.
        """
        B = tokenized_prompt.shape[0]

        if self.forced_cluster_id is not None:
            gate = jax.nn.one_hot(
                jnp.full((B,), self.forced_cluster_id, dtype=jnp.int32), self.num_loras
            )
            # Return the forced gate as both gate and probs so aux loss is a no-op.
            return gate, gate, jnp.zeros_like(gate)

        token_embeds = self.PaliGemma.llm(tokenized_prompt, method="embed").astype(jnp.float32)
        mask = tokenized_prompt_mask.astype(jnp.float32)[..., None]
        pooled = (token_embeds * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)
        # The LLM and its embedding table are frozen. Prevent the router's
        # backward pass from flowing into them — it would be silently discarded
        # by the freeze filter anyway, but the scatter-add through a
        # (vocab_size, hidden) embedding is a real memory cost and has been
        # observed to trigger CUDA_ERROR_ILLEGAL_ADDRESS near the memory ceiling.
        pooled = jax.lax.stop_gradient(pooled)

        gate, probs, logits = self.router(
            pooled, train=train, rng=rng, temperature=self.router_temperature
        )
        return gate, probs, logits

    def _router_aux_loss(
        self, probs: jax.Array, logits: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Switch-Transformer load-balancing + z-loss plus diagnostics.

        Returns (aux_loss_scalar, stats_dict). The dict uses the same keys on
        every call (including the forced-cluster debug path) so downstream
        loggers never see missing fields.

        Key metrics (all float32 scalars):
          router/balance_loss       — N * sum(f * P). At uniform: 1.0. Collapse: N.
          router/z_loss             — mean(logsumexp(logits)^2). Numerical health.
          router/per_sample_entropy — mean per-row entropy of probs.
              Starts near log(N). Low => decisive routing. If still ~log(N)
              after 5k steps, the router isn't differentiating tasks.
          router/usage_entropy      — entropy of the batch-mean distribution P.
              Starts near log(N). Low => concentrated usage across the batch.
              Critical signal: if this drops to 0, the router collapsed onto
              one expert (same failure mode as the centroid pipeline had).
          router/dead_experts       — count of experts with hard-usage < 1/(2N).
              If > 0 and stays positive for >2k steps: raise balance weight.
          router/hard_usage_min/max — tightest indicator of imbalance.
              hard_usage_max == 1.0 means total collapse THIS batch.
          router/hard_usage_{i}     — per-expert hard usage, one scalar per LoRA.
          router/max_abs_logit      — if this climbs above ~30, raise z-loss.
        """
        num_loras = self.num_loras
        zero = jnp.asarray(0.0, dtype=jnp.float32)

        if self.forced_cluster_id is not None:
            unit = jnp.asarray(1.0, dtype=jnp.float32)
            stats: dict[str, jax.Array] = {
                "router/balance_loss": zero,
                "router/z_loss": zero,
                "router/per_sample_entropy": zero,
                "router/usage_entropy": zero,
                "router/dead_experts": jnp.asarray(num_loras - 1, dtype=jnp.float32),
                "router/hard_usage_min": zero,
                "router/hard_usage_max": unit,
                "router/max_abs_logit": zero,
            }
            for i in range(num_loras):
                stats[f"router/hard_usage_{i}"] = unit if i == self.forced_cluster_id else zero
            return zero, stats

        hard = jax.nn.one_hot(jnp.argmax(probs, axis=-1), num_loras)
        f = hard.mean(axis=0)                                       # (N,)
        P = probs.mean(axis=0)                                      # (N,)

        l_balance = num_loras * jnp.sum(f * P)
        l_z = jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2)
        aux = (
            self.router_balance_weight * l_balance
            + self.router_z_loss_weight * l_z
        ).astype(jnp.float32)

        # --- Diagnostics ---
        eps = 1e-9
        per_sample_entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + eps), axis=-1))
        usage_entropy = -jnp.sum(P * jnp.log(P + eps))
        dead_thresh = 0.5 / num_loras                               # half of uniform
        dead_experts = jnp.sum((f < dead_thresh).astype(jnp.float32))

        stats = {
            "router/balance_loss": l_balance.astype(jnp.float32),
            "router/z_loss": l_z.astype(jnp.float32),
            "router/per_sample_entropy": per_sample_entropy.astype(jnp.float32),
            "router/usage_entropy": usage_entropy.astype(jnp.float32),
            "router/dead_experts": dead_experts.astype(jnp.float32),
            "router/hard_usage_min": jnp.min(f).astype(jnp.float32),
            "router/hard_usage_max": jnp.max(f).astype(jnp.float32),
            "router/max_abs_logit": jnp.max(jnp.abs(logits)).astype(jnp.float32),
        }
        for i in range(num_loras):
            stats[f"router/hard_usage_{i}"] = f[i].astype(jnp.float32)

        return aux, stats

    # ------------------------------------------------------------------
    # Forward halves
    # ------------------------------------------------------------------
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask, ar_mask, tokens = [], [], []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions,
        timestep: at.Float[at.Array, " b"],
        gate: Optional[jax.Array] = None,
        explicit_action_reason: Optional[jax.Array] = None,
        implicit_action_reason: Optional[jax.Array] = None,
        suf_type: str = "reasoner",
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask, ar_mask, tokens = [], [], []

        if suf_type == "reasoner":
            action_tokens = self.coarse_action_in_proj(noisy_actions, gate)
            time_emb = posemb_sincos(
                timestep, self.coarse_action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
            time_emb = self.coarse_time_mlp_in(time_emb, gate)
            time_emb = nnx.swish(time_emb)
            time_emb = self.coarse_time_mlp_out(time_emb, gate)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb

        elif suf_type == "expert":
            action_tokens = self.action_in_proj(noisy_actions, gate)
            time_emb = posemb_sincos(
                timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
            time_emb = self.time_mlp_in(time_emb, gate)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb, gate)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb

            if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
                exp_tokens = self.coarse_action_in_proj(explicit_action_reason, gate)
                aligned_exp = self.explicit_action_reasoner(action_expert_tokens, exp_tokens, gate=gate)

                im_tokens = implicit_action_reason
                aligned_im = self.implicit_action_reasoner_interact(
                    action_expert_tokens, im_tokens, gate=gate
                )

                x_exp = jnp.concatenate([action_expert_tokens, aligned_exp], axis=-1)
                x_exp = self.explicit_action_reason_proj(x_exp, gate)
                x_im = jnp.concatenate([action_expert_tokens, aligned_im], axis=-1)
                x_im = self.implicit_action_reason_proj(x_im, gate)

                fused = jnp.concatenate([x_exp, x_im], axis=-1)
                action_expert_tokens = self.action_reasoning_fusion(fused, fused, gate=gate)

            elif self.adopt_explicit_action_reasoner:
                exp_tokens = self.coarse_action_in_proj(explicit_action_reason, gate)
                aligned_exp = self.explicit_action_reasoner(action_expert_tokens, exp_tokens, gate=gate)
                action_expert_tokens = jnp.concatenate([action_expert_tokens, aligned_exp], axis=-1)
                action_expert_tokens = self.action_reasoning_fusion(action_expert_tokens, gate)

            elif self.adopt_implicit_action_reasoner:
                aligned_im = self.implicit_action_reasoner_interact(
                    action_expert_tokens, implicit_action_reason, gate=gate
                )
                action_expert_tokens = jnp.concatenate([action_expert_tokens, aligned_im], axis=-1)
                action_expert_tokens = self.action_reasoning_fusion(action_expert_tokens, gate)
        else:
            raise ValueError(f"Unknown suffix type: {suf_type}")

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        if suf_type == "reasoner":
            ar_mask += [True] + ([False] * (self.coarse_action_horizon - 1))
        else:
            ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # ------------------------------------------------------------------
    # Losses / sampling
    # ------------------------------------------------------------------
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions,
        *,
        train: bool = False,
    ):
        preprocess_rng, time_rng, coarse_noise_rng, expert_noise_rng, router_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]

        # Compute the gate via the learned router. During training we pass a
        # fresh rng so Gumbel noise breaks initial symmetry among experts.
        gate, router_probs, router_logits = self._compute_gate(
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            rng=router_rng,
            train=train,
        )
        aux_loss, router_stats = self._router_aux_loss(router_probs, router_logits)

        coarse_noise = jax.random.normal(coarse_noise_rng, coarse_actions.shape)
        expert_noise = jax.random.normal(expert_noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        x_ref_t = time_expanded * coarse_noise + (1.0 - time_expanded) * coarse_actions
        u_ref_t = coarse_noise - coarse_actions
        x_expert_t = time_expanded * expert_noise + (1.0 - time_expanded) * actions
        u_expert_t = expert_noise - actions

        # Shared prefix forward
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions_prefix
        )

        # ---- Explicit (EAR) branch ----
        if self.adopt_explicit_action_reasoner:
            suffix_ref_tokens, suffix_ref_mask, suffix_ref_ar_mask, adarms_ref = self.embed_suffix(
                observation, x_ref_t, time, gate=gate, suf_type="reasoner"
            )
            input_mask = jnp.concatenate([prefix_mask, suffix_ref_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ref_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_ref_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_ref_tokens, None],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_ref, None],
            )
            # Teacher forcing: feed ground-truth coarse actions into the expert.
            explicit_action_reason = coarse_actions
        else:
            suffix_ref_out = None
            explicit_action_reason = None

        # ---- Implicit (IAR) branch ----
        if self.adopt_implicit_action_reasoner:
            K_all, V_all = kv_cache
            K_r = einops.rearrange(K_all, "L B T 1 D -> B L T D")
            V_r = einops.rearrange(V_all, "L B T 1 D -> B L T D")
            implicit_action_reason = self.implicit_action_reasoner(K_r, V_r, gate=gate)
        else:
            implicit_action_reason = None

        # ---- Expert (action-head) branch ----
        suffix_expert_tokens, suffix_expert_mask, suffix_expert_ar_mask, adarms_expert = self.embed_suffix(
            observation,
            x_expert_t,
            time,
            gate=gate,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suf_type="expert",
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_expert_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_expert_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, _, suffix_expert_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None, suffix_expert_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None, adarms_expert],
        )

        if self.adopt_explicit_action_reasoner:
            v_ref_t = self.coarse_action_out_proj(
                suffix_ref_out[:, -self.coarse_action_horizon:], gate
            )
            v_expert_t = self.action_out_proj(
                suffix_expert_out[:, -self.action_horizon:], gate
            )
            flow_loss = (
                jnp.mean(jnp.square(u_ref_t - v_ref_t))
                + jnp.mean(jnp.square(u_expert_t - v_expert_t))
            )
            total = flow_loss + aux_loss
            stats = {
                **router_stats,
                "loss/flow": flow_loss.astype(jnp.float32),
                "loss/aux": aux_loss.astype(jnp.float32),
            }
            return total, stats

        v_expert_t = self.action_out_proj(suffix_expert_out[:, -self.action_horizon:], gate)
        flow_loss = jnp.mean(jnp.square(u_expert_t - v_expert_t))
        total = flow_loss + aux_loss
        stats = {
            **router_stats,
            "loss/flow": flow_loss.astype(jnp.float32),
            "loss/aux": aux_loss.astype(jnp.float32),
        }
        return total, stats

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        gate, _, _ = self._compute_gate(
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            rng=None,
            train=False,
        )

        ref_rng, expert_rng = jax.random.split(rng, 2)
        ref_noise = jax.random.normal(ref_rng, (batch_size, self.coarse_action_horizon, self.action_dim))
        expert_noise = jax.random.normal(expert_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions
        )

        if self.adopt_implicit_action_reasoner:
            K_all, V_all = kv_cache
            K_r = einops.rearrange(K_all, "L B T 1 D -> B L T D")
            V_r = einops.rearrange(V_all, "L B T 1 D -> B L T D")
            implicit_action_reason = self.implicit_action_reasoner(K_r, V_r, gate=gate)
        else:
            implicit_action_reason = None

        # EAR refinement loop
        def step_ref(carry):
            x_t, t, step_idx = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(t, batch_size), gate=gate, suf_type="reasoner"
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out, _), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None],
                mask=full_attn,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond, None],
            )
            v_t = self.coarse_action_out_proj(suffix_out[:, -self.coarse_action_horizon:], gate)
            return x_t + dt * v_t, t + dt, step_idx + 1

        def cond_ref(carry):
            _, t, _ = carry
            return t >= -dt / 2

        if self.adopt_explicit_action_reasoner:
            explicit_action_reason, _, _ = jax.lax.while_loop(cond_ref, step_ref, (ref_noise, 1.0, 1))
        else:
            explicit_action_reason = None

        # Expert refinement loop
        def step_expert(carry):
            x_t, t, step_idx = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(t, batch_size),
                gate=gate,
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
                suf_type="expert",
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, _, suffix_out), _ = self.PaliGemma.llm(
                [None, None, suffix_tokens],
                mask=full_attn,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:], gate)
            return x_t + dt * v_t, t + dt, step_idx + 1

        def cond_expert(carry):
            _, t, _ = carry
            return t >= -dt / 2

        x_final, _, _ = jax.lax.while_loop(cond_expert, step_expert, (expert_noise, 1.0, 1))

        if self.adopt_explicit_action_reasoner:
            return {"actions": x_final, "coarse_actions": explicit_action_reason}
        return {"actions": x_final}