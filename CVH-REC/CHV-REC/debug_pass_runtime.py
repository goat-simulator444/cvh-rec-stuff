import gc
import importlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_DIR = "/home/abhinav24/Desktop/cvh-rec/CVH-REC/CHV-REC"
DATA_PATH = "dataset/"
DATASET = "fold1"


def _clear_modules() -> None:
    for name in ["main", "utility.batch_test", "utility.parser"]:
        if name in sys.modules:
            del sys.modules[name]


def build_engine(extra_args: List[str]):
    os.chdir(PROJECT_DIR)
    sys.argv = ["main.py"] + extra_args
    _clear_modules()
    main = importlib.import_module("main")

    config = {
        "n_users": main.data_generator.n_users,
        "n_items": main.data_generator.n_items,
        "n_tags": main.data_generator.n_tags,
    }
    engine = main.Model_Wrapper(data_config=config, pretrain_data=None)
    return main, engine


def summarize_outputs(engine, outputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    report = {
        "num_outputs": len(outputs),
        "n_users": int(engine.n_mashup),
        "n_items": int(engine.n_api),
        "embed_dim": int(engine.emb_dim),
        "tensor_shapes": {},
        "nan_check": {},
    }

    names = [
        "m_embeddings",
        "a_embeddings",
        "m_call_proj",
        "m_tag_proj",
        "a_call_proj",
        "a_tag_proj",
        "m_local_proj",
        "a_local_proj",
        "m_global_proj",
        "a_global_proj",
        "mashup_tag_logits",
        "api_tag_logits",
    ]

    for i, name in enumerate(names):
        if i >= len(outputs):
            break
        t = outputs[i]
        if torch.is_tensor(t):
            report["tensor_shapes"][name] = list(t.shape)
            report["nan_check"][name] = bool(torch.isfinite(t).all().item())

    return report


def _batch_losses(engine, users, pos_items, neg_items):
    def _to_float(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu())
        return float(x)

    outputs = engine._forward_hgnn()
    if len(outputs) < 12:
        raise RuntimeError(f"Expected >=12 outputs, got {len(outputs)}")

    (
        m_embeddings,
        a_embeddings,
        m_c_proj,
        m_t_proj,
        a_c_proj,
        a_t_proj,
        m_local_proj,
        a_local_proj,
        m_global_proj,
        a_global_proj,
        mashup_tag_logits,
        api_tag_logits,
    ) = outputs[:12]

    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    batch_mashup_tag_loss = bce_loss_fn(mashup_tag_logits, engine.mashup_tag_labels.to(mashup_tag_logits.device))
    batch_api_tag_loss = bce_loss_fn(api_tag_logits, engine.api_tag_labels.to(api_tag_logits.device))

    batch_edge_diff_loss = torch.tensor(0.0, device=engine.device)
    edge_pairs = 0
    if engine.edge_diffusion is not None and engine.edge_diffusion_weight > 0.0:
        edge_m_embeddings = m_embeddings.detach() if engine.edge_diffusion_weight > 0 else m_embeddings
        edge_a_embeddings = a_embeddings.detach() if engine.edge_diffusion_weight > 0 else a_embeddings
        batch_edge_diff_loss, edge_stats = engine.edge_diffusion.edge_loss(
            mashup_embeddings=edge_m_embeddings,
            api_embeddings=edge_a_embeddings,
            real_edge_index=engine.real_edge_index,
            negative_ratio=1.0,
            max_pos_edges=2048,
        )
        edge_pairs = int(edge_stats.get("num_pairs", 0))

    u_g_embeddings = m_embeddings[users]
    pos_i_g_embeddings = a_embeddings[pos_items]
    neg_i_g_embeddings = a_embeddings[neg_items]

    batch_mf_loss, batch_emb_loss, batch_reg_loss = engine.bpr_loss(
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
    )

    batch_mashup_cl_loss = engine.contrastive_loss(m_c_proj, m_t_proj, temperature=0.5)
    batch_api_cl_loss = engine.contrastive_loss(a_c_proj, a_t_proj, temperature=0.5)
    batch_mashup_global_cl_loss = engine.contrastive_loss(m_local_proj, m_global_proj, temperature=0.5)
    batch_api_global_cl_loss = engine.contrastive_loss(a_local_proj, a_global_proj, temperature=0.5)

    cl_loss_weight = 0.1

    batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss
    batch_loss += cl_loss_weight * batch_mashup_cl_loss + cl_loss_weight * batch_api_cl_loss
    batch_loss += cl_loss_weight * batch_mashup_global_cl_loss + cl_loss_weight * batch_api_global_cl_loss
    batch_loss += batch_mashup_tag_loss + batch_api_tag_loss
    batch_loss += engine.edge_diffusion_weight * batch_edge_diff_loss

    parts = {
        "total": _to_float(batch_loss),
        "mf": _to_float(batch_mf_loss),
        "emb": _to_float(batch_emb_loss),
        "reg": _to_float(batch_reg_loss),
        "m_cl": _to_float(batch_mashup_cl_loss),
        "a_cl": _to_float(batch_api_cl_loss),
        "m_g_cl": _to_float(batch_mashup_global_cl_loss),
        "a_g_cl": _to_float(batch_api_global_cl_loss),
        "m_tag": _to_float(batch_mashup_tag_loss),
        "a_tag": _to_float(batch_api_tag_loss),
        "edge_diff": _to_float(batch_edge_diff_loss),
        "edge_pairs": int(edge_pairs),
    }

    return batch_loss, outputs, parts


def gradient_report(module: torch.nn.Module) -> Dict[str, Any]:
    if module is None:
        return {"present": False}

    total_params = 0
    grad_params = 0
    finite_grad_params = 0
    nonzero_grad_params = 0

    for p in module.parameters():
        total_params += 1
        if p.grad is None:
            continue
        grad_params += 1
        if torch.isfinite(p.grad).all().item():
            finite_grad_params += 1
        if torch.count_nonzero(p.grad).item() > 0:
            nonzero_grad_params += 1

    return {
        "present": True,
        "total_params": total_params,
        "grad_params": grad_params,
        "finite_grad_params": finite_grad_params,
        "nonzero_grad_params": nonzero_grad_params,
    }


def run_single_grad_check(main, engine) -> Dict[str, Any]:
    engine.model.train()
    if engine.temporal_model is not None:
        engine.temporal_model.train()
    if engine.edge_diffusion is not None:
        engine.edge_diffusion.train()

    engine.optimizer.zero_grad(set_to_none=True)

    users, pos_items, neg_items = main.data_generator.sample()
    if len(users) == 0:
        raise RuntimeError("Sampled empty batch")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=engine.device)
        torch.cuda.synchronize(device=engine.device)

    t0 = time.perf_counter()
    batch_loss, outputs, parts = _batch_losses(engine, users, pos_items, neg_items)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=engine.device)
    t1 = time.perf_counter()

    batch_loss.backward()

    if engine.temporal_model is not None:
        engine.temporal_model.clip_gradients(max_norm=5.0)
    else:
        torch.nn.utils.clip_grad_norm_(engine.model.parameters(), max_norm=5.0)

    if engine.edge_diffusion is not None:
        torch.nn.utils.clip_grad_norm_(engine.edge_diffusion.parameters(), max_norm=5.0)

    report = {
        "loss_parts": parts,
        "forward_backward_time_sec": t1 - t0,
        "is_finite_loss": bool(torch.isfinite(batch_loss).item()),
        "model_grads": gradient_report(engine.model),
        "temporal_grads": gradient_report(engine.temporal_model),
        "diffusion_grads": gradient_report(engine.edge_diffusion),
        "output_nan_checks": summarize_outputs(engine, outputs)["nan_check"],
    }

    if torch.cuda.is_available():
        report["cuda_memory"] = {
            "allocated_mb": torch.cuda.memory_allocated(engine.device) / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved(engine.device) / (1024 ** 2),
            "peak_allocated_mb": torch.cuda.max_memory_allocated(engine.device) / (1024 ** 2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(engine.device) / (1024 ** 2),
        }

    engine.optimizer.step()
    return report


def run_small_training(main, engine, steps: int = 24) -> Dict[str, Any]:
    losses = []
    finite_embeddings = True

    engine.model.train()
    if engine.temporal_model is not None:
        engine.temporal_model.train()
    if engine.edge_diffusion is not None:
        engine.edge_diffusion.train()

    for _ in range(steps):
        users, pos_items, neg_items = main.data_generator.sample()
        if len(users) == 0:
            continue

        engine.optimizer.zero_grad(set_to_none=True)
        batch_loss, outputs, _ = _batch_losses(engine, users, pos_items, neg_items)

        if len(outputs) >= 2:
            finite_embeddings = finite_embeddings and bool(torch.isfinite(outputs[0]).all().item())
            finite_embeddings = finite_embeddings and bool(torch.isfinite(outputs[1]).all().item())

        if not torch.isfinite(batch_loss):
            losses.append(float("nan"))
            continue

        batch_loss.backward()

        if engine.temporal_model is not None:
            engine.temporal_model.clip_gradients(max_norm=5.0)
        else:
            torch.nn.utils.clip_grad_norm_(engine.model.parameters(), max_norm=5.0)
        if engine.edge_diffusion is not None:
            torch.nn.utils.clip_grad_norm_(engine.edge_diffusion.parameters(), max_norm=5.0)

        engine.optimizer.step()
        losses.append(float(batch_loss.detach().cpu()))

    finite_losses = [x for x in losses if math.isfinite(x)]
    trend = None
    if len(finite_losses) >= 2:
        trend = finite_losses[-1] - finite_losses[0]

    return {
        "steps": steps,
        "logged_losses": losses,
        "finite_embeddings": finite_embeddings,
        "finite_loss_count": len(finite_losses),
        "loss_delta_last_minus_first": trend,
        "loss_decreased": (trend is not None and trend < 0),
    }


def compare_static_vs_new(static_outputs, new_outputs) -> Dict[str, Any]:
    ms, as_ = static_outputs[0], static_outputs[1]
    mn, an = new_outputs[0], new_outputs[1]

    def _stats(t):
        return {
            "shape": list(t.shape),
            "mean": float(t.mean().detach().cpu()),
            "std": float(t.std().detach().cpu()),
            "norm_mean": float(torch.norm(t, dim=1).mean().detach().cpu()),
            "finite": bool(torch.isfinite(t).all().item()),
        }

    overlap_users = min(ms.size(0), mn.size(0), 512)
    overlap_items = min(as_.size(0), an.size(0), 512)

    ms_sub = ms[:overlap_users]
    mn_sub = mn[:overlap_users]
    as_sub = as_[:overlap_items]
    an_sub = an[:overlap_items]

    cosine_user = torch.nn.functional.cosine_similarity(ms_sub, mn_sub, dim=1).mean()
    cosine_item = torch.nn.functional.cosine_similarity(as_sub, an_sub, dim=1).mean()

    score_static = torch.matmul(ms_sub[:128], as_sub[:128].T)
    score_new = torch.matmul(mn_sub[:128], an_sub[:128].T)
    score_l2 = torch.norm(score_static - score_new).item() / max(1, score_static.numel())

    return {
        "static_mashup": _stats(ms),
        "new_mashup": _stats(mn),
        "static_api": _stats(as_),
        "new_api": _stats(an),
        "mean_cosine_user": float(cosine_user.detach().cpu()),
        "mean_cosine_item": float(cosine_item.detach().cpu()),
        "score_l2_per_entry": float(score_l2),
    }


def estimate_dense_memory_mb(n_users: int, n_items: int, dtype_bytes: int = 4) -> float:
    return (n_users * n_items * dtype_bytes) / (1024 ** 2)


def main():
    base_args = [
        "--data_path", DATA_PATH,
        "--dataset", DATASET,
        "--batch_size", "128",
        "--layer_num", "2",
        "--save_flag", "0",
        "--verbose", "0",
        "--epoch", "2",
        "--temporal_debug_logs", "0",
        "--edge_gen_refresh_interval", "0",
        "--edge_gen_warmup_epochs", "1000",
    ]

    static_args = base_args + [
        "--temporal_hypergraph", "0",
        "--use_temporal", "0",
        "--use_diffusion", "0",
        "--edge_diffusion", "0",
    ]

    new_args = base_args + [
        "--temporal_hypergraph", "1",
        "--use_temporal", "1",
        "--temporal_window_size", "256",
        "--temporal_max_steps", "4",
        "--temporal_stride", "1",
        "--use_diffusion", "1",
        "--edge_diffusion", "1",
        "--edge_diffusion_weight", "0.05",
        "--edge_diffusion_max_pos_per_batch", "512",
        "--edge_gen_chunk_size", "512",
    ]

    report = {
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    }

    print("[1/5] Building static engine...")
    static_main, static_engine = build_engine(static_args)
    static_engine.model.eval()
    with torch.no_grad():
        static_outputs = static_engine._forward_hgnn()
    report["static_forward"] = summarize_outputs(static_engine, static_outputs)

    print("[2/5] Static gradient and mini-train checks...")
    report["static_gradient"] = run_single_grad_check(static_main, static_engine)
    report["static_small_train"] = run_small_training(static_main, static_engine, steps=20)

    del static_main
    del static_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[3/5] Building temporal+diffusion engine...")
    new_main, new_engine = build_engine(new_args)
    new_engine.model.eval()
    if new_engine.temporal_model is not None:
        new_engine.temporal_model.eval()
    with torch.no_grad():
        new_outputs = new_engine._forward_hgnn()
    report["new_forward"] = summarize_outputs(new_engine, new_outputs)

    print("[4/5] New model gradient and mini-train checks...")
    report["new_gradient"] = run_single_grad_check(new_main, new_engine)
    report["new_small_train"] = run_small_training(new_main, new_engine, steps=20)

    print("[5/5] Comparing static vs new outputs and memory risk...")
    report["compare_static_vs_new"] = compare_static_vs_new(static_outputs, new_outputs)
    report["memory_risk"] = {
        "n_users": int(new_engine.n_mashup),
        "n_items": int(new_engine.n_api),
        "dense_call_matrix_mb": estimate_dense_memory_mb(new_engine.n_mashup, new_engine.n_api),
        "temporal_steps": int(new_engine.last_temporal_step_count),
        "temporal_edge_counts": [int(x) for x in new_engine.temporal_call_edge_counts],
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
