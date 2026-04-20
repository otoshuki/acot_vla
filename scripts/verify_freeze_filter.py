"""Verify the multi-LoRA freeze filter does what we expect.

No evaluation, no rollouts — pure param-accounting and a dummy optimizer step.
Run this BEFORE launching the real training to catch filter bugs.

Usage:
    python scripts/verify_freeze_filter.py --config-name teamace_multilora_taskgroups

Exits 0 on success, non-zero with an explanation on any mismatch.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer


def _path_to_str(path_tuple) -> str:
    parts = []
    for p in path_tuple:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        else:
            parts.append(str(p))
    return "/".join(parts)


def _classify_path(path: str) -> str:
    """Bucket a param path into one of our interpretability buckets."""
    if re.search(r"PaliGemma/img|/img/", path) or path.startswith("img"):
        return "vision"
    if re.search(r"/llm/.*_1/|/llm_1/", path):
        return "llm_coarse_expert"
    if re.search(r"/llm/.*_2/|/llm_2/", path):
        return "llm_action_expert"
    if "/llm/" in path or path.startswith("llm") or "PaliGemma/llm" in path:
        return "llm_base"
    if re.match(r".*lora_[AB](_shared)?$", path):
        return "multilora_adapter"
    if path.endswith("/kernel") or path.endswith("/bias"):
        return "action_head_base"
    if "query_params" in path:
        return "query_params"
    return "other"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dummy-step", action="store_true",
                        help="Run one dummy optimizer step and verify no frozen param moved.")
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    logging.info(f"Config: {config.name}")

    rng = jax.random.key(0)
    model = config.model.create(rng)

    # Enumerate every nnx.Param and its frozen/trainable status.
    all_param_state = nnx.state(model, nnx.Param)
    trainable_state = nnx.state(model, config.trainable_filter)

    all_flat = traverse_util.flatten_dict(all_param_state.to_pure_dict())
    trainable_flat = traverse_util.flatten_dict(trainable_state.to_pure_dict())
    trainable_paths = {_path_to_str(k) for k in trainable_flat.keys()}

    bucket_totals = defaultdict(lambda: {"trainable": 0, "frozen": 0,
                                          "trainable_count": 0, "frozen_count": 0})
    failures = []

    # Invariants we expect:
    #   1. Every vision param is frozen.
    #   2. Every llm_base param is frozen (including Gemma's own LoRAs).
    #   3. Every llm_coarse_expert and llm_action_expert param is frozen.
    #   4. Every multilora_adapter is trainable.
    #   5. Every action_head_base (kernel/bias outside LLM/vision) is frozen.
    #   6. Every query_params is frozen.
    expected = {
        "vision": "frozen",
        "llm_base": "frozen",
        "llm_coarse_expert": "frozen",
        "llm_action_expert": "frozen",
        "multilora_adapter": "trainable",
        "action_head_base": "frozen",
        "query_params": "frozen",
    }

    for path_tuple, arr in all_flat.items():
        path = _path_to_str(path_tuple)
        is_trainable = path in trainable_paths
        status = "trainable" if is_trainable else "frozen"
        bucket = _classify_path(path)
        n = int(np.prod(arr.shape)) if arr.shape else 1
        bucket_totals[bucket][status] += n
        bucket_totals[bucket][f"{status}_count"] += 1

        if bucket in expected and status != expected[bucket]:
            failures.append((bucket, path, status, expected[bucket]))

    # Print summary table.
    print("\n" + "=" * 80)
    print(f"{'Bucket':<25} {'Trainable':>15} {'Frozen':>15} {'Status':>20}")
    print("-" * 80)
    for bucket in sorted(bucket_totals.keys()):
        t = bucket_totals[bucket]
        exp = expected.get(bucket, "—")
        status_str = f"expect: {exp}" if bucket in expected else "mixed (ok)"
        print(f"{bucket:<25} {t['trainable']:>15,} {t['frozen']:>15,} {status_str:>20}")
    print("=" * 80)

    total_trainable = sum(t["trainable"] for t in bucket_totals.values())
    total_frozen = sum(t["frozen"] for t in bucket_totals.values())
    print(f"{'TOTAL':<25} {total_trainable:>15,} {total_frozen:>15,}")
    pct = 100.0 * total_trainable / max(total_trainable + total_frozen, 1)
    print(f"Trainable fraction: {pct:.2f}%")

    # Report failures.
    if failures:
        print("\nFAILURES:")
        by_bucket = defaultdict(list)
        for b, p, s, e in failures:
            by_bucket[b].append((p, s, e))
        for bucket, items in by_bucket.items():
            print(f"  [{bucket}] {len(items)} paths in wrong bucket (showing first 5):")
            for p, s, e in items[:5]:
                print(f"      got={s} want={e}  {p}")
        print(f"\n{len(failures)} invariant violations. See above.")
        sys.exit(1)

    print("\nAll invariants satisfied.")

    if args.dummy_step:
        print("\nRunning dummy optimizer step to verify no frozen param moves...")
        tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
        params_before = nnx.state(model, nnx.Param).to_pure_dict()
        trainable_params = nnx.state(model, config.trainable_filter)
        opt_state = tx.init(trainable_params)

        # Fake gradient: ones_like for every trainable param (guaranteed nonzero update).
        fake_grads = jax.tree.map(lambda x: jnp.ones_like(x), trainable_params)
        updates, _ = tx.update(fake_grads, opt_state, trainable_params)
        new_trainable = optax.apply_updates(trainable_params, updates)
        nnx.update(model, new_trainable)

        params_after = nnx.state(model, nnx.Param).to_pure_dict()
        before_flat = traverse_util.flatten_dict(params_before)
        after_flat = traverse_util.flatten_dict(params_after)

        moved_frozen = []
        for key in before_flat:
            path = _path_to_str(key)
            if path in trainable_paths:
                continue
            diff = float(jnp.max(jnp.abs(before_flat[key] - after_flat[key])))
            if diff > 0.0:
                moved_frozen.append((path, diff))

        if moved_frozen:
            print(f"\nFROZEN PARAMS MOVED after dummy step ({len(moved_frozen)} total):")
            for p, d in moved_frozen[:10]:
                print(f"    diff={d:.3e}  {p}")
            sys.exit(2)
        print("Confirmed: no frozen param moved after dummy optimizer step.")


if __name__ == "__main__":
    main()