import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Deterministic time embedding used by the diffusion scorer."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.dim // 2
        if half_dim <= 0:
            return timesteps.unsqueeze(-1)

        freq_factor = -math.log(10000.0) / max(half_dim - 1, 1)
        exponents = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * freq_factor
        )
        angles = timesteps.unsqueeze(1) * exponents.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class EdgeDiffusion(nn.Module):
    """
    Diffusion-inspired edge generator.

    Input:
        mashup embedding and API embedding for each candidate pair.
    Output:
        probability that the pair forms an edge.

    Training:
        positives come from real graph edges,
        negatives are sampled from non-edge pairs.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        diffusion_steps: int = 8,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.diffusion_steps = max(2, int(diffusion_steps))

        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        pair_feature_dim = self.embedding_dim * 4
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._neg_sampling_cache = {}

    @staticmethod
    def _validate_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        return edge_index.long()

    def _pair_features(self, mashup_emb: torch.Tensor, api_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                mashup_emb,
                api_emb,
                torch.abs(mashup_emb - api_emb),
                mashup_emb * api_emb,
            ],
            dim=-1,
        )

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)

    def _q_sample(self, clean_rep: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.alpha_bars[timesteps].unsqueeze(1)
        noise = torch.randn_like(clean_rep)
        noisy_rep = torch.sqrt(alpha_bar_t) * clean_rep + torch.sqrt(1.0 - alpha_bar_t) * noise
        return noisy_rep, noise

    def forward(
        self,
        mashup_emb: torch.Tensor,
        api_emb: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        add_noise: bool = True,
    ) -> torch.Tensor:
        if mashup_emb.shape != api_emb.shape:
            raise ValueError(
                f"mashup_emb and api_emb must have the same shape, got {mashup_emb.shape} vs {api_emb.shape}"
            )

        batch_size = mashup_emb.size(0)
        device = mashup_emb.device

        if timesteps is None:
            timesteps = self._sample_timesteps(batch_size, device)
        else:
            if timesteps.dim() == 0:
                timesteps = timesteps.repeat(batch_size)
            timesteps = timesteps.long().to(device)
            if timesteps.numel() != batch_size:
                raise ValueError(
                    f"timesteps must have {batch_size} elements, got {timesteps.numel()}"
                )
            timesteps = torch.clamp(timesteps, min=0, max=self.diffusion_steps - 1)

        pair_rep = self.pair_encoder(self._pair_features(mashup_emb, api_emb))

        if add_noise:
            noisy_rep, _ = self._q_sample(pair_rep, timesteps)
        else:
            noisy_rep = pair_rep

        t_rep = self.time_embed(timesteps)
        logits = self.edge_head(torch.cat([noisy_rep, t_rep], dim=-1)).squeeze(-1)
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        if return_logits:
            return logits

        return torch.sigmoid(logits)

    def sample_negative_edges(
        self,
        real_edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        real_edge_index = self._validate_edge_index(real_edge_index)
        if num_samples <= 0 or num_users <= 0 or num_items <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        with torch.no_grad():
            cache_key = (
                int(real_edge_index.data_ptr()),
                int(num_items),
                int(real_edge_index.size(1)),
            )
            positive_flat = self._neg_sampling_cache.get(cache_key)
            if positive_flat is None:
                positive_flat = set(
                    (real_edge_index[0].detach().cpu() * num_items + real_edge_index[1].detach().cpu()).tolist()
                )
                # Keep only the latest cache entry to avoid unbounded growth.
                self._neg_sampling_cache = {cache_key: positive_flat}

            total_possible = num_users * num_items - len(positive_flat)
            if total_possible <= 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)

            target_samples = min(int(num_samples), total_possible)
            sampled_flat = set()
            sampled_users = []
            sampled_items = []

            batch_size = max(1024, min(8192, target_samples * 2))
            max_trials = target_samples * 25 + 1000
            trials = 0

            while len(sampled_flat) < target_samples and trials < max_trials:
                cand_users = torch.randint(0, num_users, (batch_size,), device=device)
                cand_items = torch.randint(0, num_items, (batch_size,), device=device)

                users_list = cand_users.detach().cpu().tolist()
                items_list = cand_items.detach().cpu().tolist()

                for u, i in zip(users_list, items_list):
                    flat = u * num_items + i
                    if flat in positive_flat or flat in sampled_flat:
                        continue
                    sampled_flat.add(flat)
                    sampled_users.append(u)
                    sampled_items.append(i)
                    if len(sampled_flat) >= target_samples:
                        break

                trials += 1

            if not sampled_users:
                return torch.empty((2, 0), dtype=torch.long, device=device)

            return torch.tensor([sampled_users, sampled_items], dtype=torch.long, device=device)

    def edge_loss(
        self,
        mashup_embeddings: torch.Tensor,
        api_embeddings: torch.Tensor,
        real_edge_index: torch.Tensor,
        negative_ratio: float = 1.0,
        max_pos_edges: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        device = mashup_embeddings.device
        real_edge_index = self._validate_edge_index(real_edge_index.to(device))

        if real_edge_index.numel() == 0:
            zero = torch.zeros((), device=device)
            return zero, {
                "num_pos": 0,
                "num_neg": 0,
                "num_pairs": 0,
            }

        pos_edge_index = real_edge_index
        if max_pos_edges is not None and max_pos_edges > 0 and pos_edge_index.size(1) > max_pos_edges:
            perm = torch.randperm(pos_edge_index.size(1), device=device)[: int(max_pos_edges)]
            pos_edge_index = pos_edge_index[:, perm]

        num_pos = pos_edge_index.size(1)
        num_neg = max(1, int(math.ceil(max(0.0, negative_ratio) * num_pos)))

        neg_edge_index = self.sample_negative_edges(
            real_edge_index=real_edge_index,
            num_users=mashup_embeddings.size(0),
            num_items=api_embeddings.size(0),
            num_samples=num_neg,
            device=device,
        )

        if neg_edge_index.numel() == 0:
            pair_index = pos_edge_index
            labels = torch.ones(num_pos, device=device)
        else:
            pair_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat(
                [
                    torch.ones(num_pos, device=device),
                    torch.zeros(neg_edge_index.size(1), device=device),
                ],
                dim=0,
            )

        mashup_pair_emb = mashup_embeddings[pair_index[0]]
        api_pair_emb = api_embeddings[pair_index[1]]

        logits = self.forward(mashup_pair_emb, api_pair_emb, return_logits=True, add_noise=True)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        bce_loss = torch.nan_to_num(bce_loss, nan=0.0, posinf=100.0, neginf=100.0)

        return bce_loss, {
            "num_pos": int(num_pos),
            "num_neg": int(neg_edge_index.size(1)),
            "num_pairs": int(pair_index.size(1)),
        }

    def _score_user_against_all_items(
        self,
        user_embedding: torch.Tensor,
        api_embeddings: torch.Tensor,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        probs = []
        item_count = api_embeddings.size(0)

        for start in range(0, item_count, chunk_size):
            end = min(start + chunk_size, item_count)
            api_chunk = api_embeddings[start:end]
            user_chunk = user_embedding.unsqueeze(0).expand(end - start, -1)
            t0 = torch.zeros(end - start, device=api_embeddings.device, dtype=torch.long)
            chunk_probs = self.forward(
                user_chunk,
                api_chunk,
                timesteps=t0,
                return_logits=False,
                add_noise=False,
            )
            probs.append(chunk_probs)

        return torch.cat(probs, dim=0) if probs else torch.empty(0, device=api_embeddings.device)

    @staticmethod
    def _deduplicate_with_max_score(
        edge_index: torch.Tensor,
        edge_scores: torch.Tensor,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return edge_index, edge_scores

        edge_index = edge_index.long()
        flat_ids = (edge_index[0] * num_items + edge_index[1]).detach().cpu().tolist()
        score_values = edge_scores.detach().cpu().tolist()
        user_values = edge_index[0].detach().cpu().tolist()
        item_values = edge_index[1].detach().cpu().tolist()

        dedup = {}
        for flat_id, score, user_id, item_id in zip(flat_ids, score_values, user_values, item_values):
            if flat_id not in dedup or score > dedup[flat_id][2]:
                dedup[flat_id] = (user_id, item_id, score)

        users, items, scores = [], [], []
        for user_id, item_id, score in dedup.values():
            users.append(user_id)
            items.append(item_id)
            scores.append(score)

        device = edge_index.device
        edge_index_out = torch.tensor([users, items], dtype=torch.long, device=device)
        score_out = torch.tensor(scores, dtype=edge_scores.dtype, device=device)
        return edge_index_out, score_out

    def generate_synthetic_edges(
        self,
        mashup_embeddings: torch.Tensor,
        api_embeddings: torch.Tensor,
        threshold: float,
        real_edge_index: Optional[torch.Tensor] = None,
        max_edges_per_node: int = 20,
        topk_per_node: Optional[int] = 256,
        max_total_edges: Optional[int] = 50000,
        threshold_step: float = 0.02,
        max_threshold_tuning_steps: int = 4,
        chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        threshold = float(max(0.0, min(1.0, threshold)))
        threshold_step = float(max(0.0, threshold_step))
        max_tune_steps = max(0, int(max_threshold_tuning_steps))

        device = mashup_embeddings.device
        num_users = mashup_embeddings.size(0)
        num_items = api_embeddings.size(0)

        if num_users == 0 or num_items == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_prob = torch.empty((0,), dtype=torch.float32, device=device)
            return empty_idx, empty_prob, {
                "threshold": threshold,
                "effective_threshold": threshold,
                "generated_edges": 0,
                "trimmed_by_budget": 0,
            }

        real_lookup = {}
        if real_edge_index is not None and real_edge_index.numel() > 0:
            real_edge_index = self._validate_edge_index(real_edge_index.to(device))
            users = real_edge_index[0].detach().cpu().tolist()
            items = real_edge_index[1].detach().cpu().tolist()
            for u, i in zip(users, items):
                real_lookup.setdefault(u, set()).add(i)

        effective_threshold = threshold
        generated_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        generated_scores = torch.empty((0,), dtype=torch.float32, device=device)

        for _ in range(max_tune_steps + 1):
            edge_users = []
            edge_items = []
            edge_scores = []

            for user_id in range(num_users):
                user_emb = mashup_embeddings[user_id]
                probs = self._score_user_against_all_items(user_emb, api_embeddings, chunk_size=chunk_size)
                if probs.numel() == 0:
                    continue

                if user_id in real_lookup and real_lookup[user_id]:
                    real_item_ids = list(real_lookup[user_id])
                    probs[real_item_ids] = 0.0

                if topk_per_node is not None and topk_per_node > 0:
                    k = min(int(topk_per_node), probs.size(0))
                    top_probs, top_items = torch.topk(probs, k=k, largest=True)
                    keep_mask = top_probs >= effective_threshold
                    candidate_items = top_items[keep_mask]
                    candidate_probs = top_probs[keep_mask]
                else:
                    candidate_items = torch.nonzero(probs >= effective_threshold, as_tuple=False).view(-1)
                    candidate_probs = probs[candidate_items]

                if candidate_items.numel() == 0:
                    continue

                if max_edges_per_node > 0 and candidate_items.numel() > max_edges_per_node:
                    top_probs, top_idx = torch.topk(candidate_probs, k=int(max_edges_per_node), largest=True)
                    candidate_items = candidate_items[top_idx]
                    candidate_probs = top_probs

                edge_users.extend([user_id] * candidate_items.numel())
                edge_items.extend(candidate_items.detach().cpu().tolist())
                edge_scores.extend(candidate_probs.detach().cpu().tolist())

            if edge_users:
                generated_edge_index = torch.tensor([edge_users, edge_items], dtype=torch.long, device=device)
                generated_scores = torch.tensor(edge_scores, dtype=torch.float32, device=device)
                generated_edge_index, generated_scores = self._deduplicate_with_max_score(
                    generated_edge_index,
                    generated_scores,
                    num_items=num_items,
                )
            else:
                generated_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                generated_scores = torch.empty((0,), dtype=torch.float32, device=device)

            if max_total_edges is None or max_total_edges <= 0 or generated_edge_index.size(1) <= max_total_edges:
                break

            effective_threshold = min(0.999, effective_threshold + threshold_step)

        trimmed_by_budget = 0
        if max_total_edges is not None and max_total_edges > 0 and generated_edge_index.size(1) > max_total_edges:
            top_probs, top_idx = torch.topk(generated_scores, k=int(max_total_edges), largest=True)
            generated_edge_index = generated_edge_index[:, top_idx]
            generated_scores = top_probs
            trimmed_by_budget = 1

        return generated_edge_index, generated_scores, {
            "threshold": threshold,
            "effective_threshold": float(effective_threshold),
            "generated_edges": int(generated_edge_index.size(1)),
            "trimmed_by_budget": int(trimmed_by_budget),
        }

    @staticmethod
    def augment_real_graph(
        real_edge_index: torch.Tensor,
        generated_edge_index: torch.Tensor,
        num_items: int,
    ) -> torch.Tensor:
        real_edge_index = EdgeDiffusion._validate_edge_index(real_edge_index)
        generated_edge_index = EdgeDiffusion._validate_edge_index(generated_edge_index)

        if generated_edge_index.numel() == 0:
            all_edges = real_edge_index
        elif real_edge_index.numel() == 0:
            all_edges = generated_edge_index
        else:
            all_edges = torch.cat([real_edge_index, generated_edge_index], dim=1)

        if all_edges.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=all_edges.device)

        flat = all_edges[0] * num_items + all_edges[1]
        unique_flat = torch.unique(flat)
        users = torch.div(unique_flat, num_items, rounding_mode="floor")
        items = unique_flat % num_items
        return torch.stack([users, items], dim=0)

    @staticmethod
    def edge_index_to_dense_matrix(
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        edge_index = EdgeDiffusion._validate_edge_index(edge_index)
        dense = torch.zeros((num_users, num_items), dtype=dtype, device=edge_index.device)
        if edge_index.numel() > 0:
            dense[edge_index[0], edge_index[1]] = 1.0
        return dense


def generate_synthetic_edges(
    edge_generator: EdgeDiffusion,
    mashup_embeddings: torch.Tensor,
    api_embeddings: torch.Tensor,
    threshold: float,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Helper wrapper to keep callsites simple."""
    return edge_generator.generate_synthetic_edges(
        mashup_embeddings=mashup_embeddings,
        api_embeddings=api_embeddings,
        threshold=threshold,
        **kwargs,
    )


def usage_example() -> Dict[str, int]:
    """
    Quick standalone usage example:
      1) Train edge generator with real edges vs sampled negatives
      2) Generate synthetic edges with threshold and max-edges safeguards
      3) Build G_augmented = G_real + generated_edges
    """
    torch.manual_seed(7)
    num_users, num_items, dim = 32, 64, 128

    mashup_embeddings = torch.randn(num_users, dim)
    api_embeddings = torch.randn(num_items, dim)

    real_edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 4, 5, 6, 7],
            [1, 2, 3, 4, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )

    edge_generator = EdgeDiffusion(embedding_dim=dim, hidden_dim=128, diffusion_steps=6)
    optimizer = torch.optim.Adam(edge_generator.parameters(), lr=1e-3)

    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = edge_generator.edge_loss(
            mashup_embeddings=mashup_embeddings,
            api_embeddings=api_embeddings,
            real_edge_index=real_edge_index,
            negative_ratio=1.0,
            max_pos_edges=128,
        )
        loss.backward()
        optimizer.step()

    generated_edge_index, _, _ = edge_generator.generate_synthetic_edges(
        mashup_embeddings=mashup_embeddings,
        api_embeddings=api_embeddings,
        threshold=0.75,
        real_edge_index=real_edge_index,
        max_edges_per_node=4,
        topk_per_node=16,
        max_total_edges=128,
    )

    augmented_edge_index = edge_generator.augment_real_graph(
        real_edge_index=real_edge_index,
        generated_edge_index=generated_edge_index,
        num_items=num_items,
    )

    return {
        "real_edges": int(real_edge_index.size(1)),
        "generated_edges": int(generated_edge_index.size(1)),
        "augmented_edges": int(augmented_edge_index.size(1)),
    }


if __name__ == "__main__":
    stats = usage_example()
    print("Edge diffusion usage example stats:", stats)
