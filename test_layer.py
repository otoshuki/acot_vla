"""Minimal standalone test of MultiLoRALinear without needing openpi installed."""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


class MultiLoRALinear(nnx.Module):
    def __init__(self, in_features, out_features, num_loras, lora_rank, *, rngs, param_dtype=jnp.float32, lora_alpha=None):
        self.num_loras = num_loras
        self.lora_rank = lora_rank
        self.in_features = in_features
        self.out_features = out_features
        alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self.scaling = alpha / float(lora_rank)
        std_w = 1.0 / (in_features ** 0.5)
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features), dtype=param_dtype) * std_w
        )
        self.bias = nnx.Param(jnp.zeros((out_features,), dtype=param_dtype))
        std_a = 1.0 / (in_features ** 0.5)
        self.lora_A = nnx.Param(
            jax.random.normal(rngs.params(), (num_loras, lora_rank, in_features), dtype=param_dtype) * std_a
        )
        self.lora_B = nnx.Param(jnp.zeros((num_loras, out_features, lora_rank), dtype=param_dtype))

    def __call__(self, x, gate: Optional[jax.Array] = None):
        kernel = self.kernel.value
        bias = self.bias.value
        base_out = jnp.matmul(x, kernel.astype(x.dtype)) + bias.astype(x.dtype)
        if gate is None:
            return base_out
        A_eff = jnp.einsum("bn,nri->bri", gate.astype(self.lora_A.value.dtype), self.lora_A.value)
        B_eff = jnp.einsum("bn,nor->bor", gate.astype(self.lora_B.value.dtype), self.lora_B.value)
        B_size = gate.shape[0]
        orig_shape = x.shape
        x_flat = x.reshape(B_size, -1, orig_shape[-1])
        Ax = jnp.einsum("bri,bti->btr", A_eff.astype(x_flat.dtype), x_flat)
        delta_flat = jnp.einsum("bor,btr->bto", B_eff.astype(Ax.dtype), Ax)
        delta = delta_flat.reshape(orig_shape[:-1] + (self.out_features,))
        return base_out + (self.scaling * delta).astype(base_out.dtype)


def _close(a, b, tol=1e-4):
    return bool(jnp.max(jnp.abs(a - b)) < tol)


# Test 1: gate=None behaves as plain Linear
rngs = nnx.Rngs(0)
m = MultiLoRALinear(16, 8, num_loras=3, lora_rank=4, rngs=rngs)
x = jax.random.normal(jax.random.key(1), (2, 5, 16))
out = m(x, gate=None)
# Reference: plain matmul with the same kernel/bias.
ref = jnp.matmul(x, m.kernel.value) + m.bias.value
assert _close(out, ref), "FAIL: gate=None"
print("[OK] gate=None behaves as plain Linear")

# Test 2: B init zero => zero delta
gate = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
out = m(x, gate=gate)
ref = jnp.matmul(x, m.kernel.value) + m.bias.value
assert _close(out, ref), "FAIL: B=0 delta"
print("[OK] B=0 init gives zero delta")

# Test 3: one-hot gate picks correct adapter
m.lora_B.value = 0.1 * jax.random.normal(jax.random.key(42), m.lora_B.value.shape)
x = jax.random.normal(jax.random.key(1), (3, 7, 16))
gate = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
out = m(x, gate=gate)
base = np.asarray(jnp.matmul(x, m.kernel.value) + m.bias.value)
A = np.asarray(m.lora_A.value)
B_ = np.asarray(m.lora_B.value)
x_np = np.asarray(x)
ref = np.empty_like(base)
for b in range(3):
    c = int(np.argmax(gate[b]))
    Ax = x_np[b] @ A[c].T
    BAx = Ax @ B_[c].T
    ref[b] = base[b] + m.scaling * BAx
diff = float(np.max(np.abs(np.asarray(out) - ref)))
assert diff < 1e-4, f"FAIL: diff={diff}"
print(f"[OK] One-hot gate routes to correct adapter (max diff = {diff:.2e})")

# Test 4: gradient isolation
m2 = MultiLoRALinear(4, 4, num_loras=3, lora_rank=2, rngs=nnx.Rngs(0))
m2.lora_B.value = 0.1 * jax.random.normal(jax.random.key(7), m2.lora_B.value.shape)
x = jnp.ones((1, 1, 4))
gate = jnp.array([[0.0, 1.0, 0.0]])

graphdef, state = nnx.split(m2)
def scalar_loss(st):
    mod = nnx.merge(graphdef, st)
    return jnp.sum(mod(x, gate=gate))

grads = jax.grad(scalar_loss)(state)
grads_B = grads.lora_B.value
grads_A = grads.lora_A.value
for inactive in [0, 2]:
    gb = float(jnp.sum(jnp.abs(grads_B[inactive])))
    ga = float(jnp.sum(jnp.abs(grads_A[inactive])))
    assert gb < 1e-8 and ga < 1e-8, f"FAIL: grads on inactive lora {inactive}: A={ga}, B={gb}"
gb1 = float(jnp.sum(jnp.abs(grads_B[1])))
assert gb1 > 1e-8, f"FAIL: no grad on active lora: B={gb1}"
print(f"[OK] Gradient isolation: only adapter 1 has non-zero grad (|grad_B[1]|={gb1:.4f})")

# Test 5: JIT compiles
@jax.jit
def jit_fwd(x, gate, state):
    mod = nnx.merge(graphdef, state)
    return mod(x, gate=gate)
out_jit = jit_fwd(x, gate, state)
out_eager = m2(x, gate=gate)
assert _close(out_jit, out_eager)
print("[OK] Forward compiles and runs under jax.jit")

# Test 6: multi-dim input (B, T1, T2, in)
m3 = MultiLoRALinear(8, 4, num_loras=2, lora_rank=3, rngs=nnx.Rngs(0))
m3.lora_B.value = 0.1 * jax.random.normal(jax.random.key(3), m3.lora_B.value.shape)
x4 = jax.random.normal(jax.random.key(9), (2, 3, 5, 8))
gate4 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
out4 = m3(x4, gate=gate4)
assert out4.shape == (2, 3, 5, 4), f"FAIL: shape {out4.shape}"
print(f"[OK] Multi-dim input handled: {x4.shape} -> {out4.shape}")

print("\nAll MultiLoRA smoke tests passed.")