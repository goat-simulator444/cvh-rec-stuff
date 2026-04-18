import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


TensorOrAny = Union[torch.Tensor, Any]
GraphStep = Union[Tuple[Any, ...], List[Any], Dict[str, Any]]


class SimpleSSMBlock(nn.Module):
    """Simple Mamba-like state update block.

    Update rule:
        h_t = tanh(W1 * h_{t-1} + W2 * h_current)
    """

    def __init__(
        self,
        embedding_dim: int,
        state_clamp_value: float = 10.0,
        dropout: float = 0.0,
        match_input_norm: bool = True,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.state_clamp_value = float(state_clamp_value)
        self.match_input_norm = bool(match_input_norm)

        self.prev_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.curr_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, prev_state: Optional[torch.Tensor], current_state: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(current_state):
            raise TypeError("current_state must be a torch.Tensor")
        if current_state.dim() != 2:
            raise ValueError(
                f"current_state must be 2D [num_nodes, embedding_dim], got {tuple(current_state.shape)}"
            )
        if current_state.size(-1) != self.embedding_dim:
            raise ValueError(
                f"current_state last dim mismatch: {current_state.size(-1)} != {self.embedding_dim}"
            )

        if prev_state is None:
            prev_state = torch.zeros_like(current_state)

        if prev_state.shape != current_state.shape:
            raise ValueError(
                f"prev_state and current_state shape mismatch: {tuple(prev_state.shape)} vs {tuple(current_state.shape)}"
            )

        mixed_state = self.prev_proj(prev_state) + self.curr_proj(current_state)
        mixed_state = torch.nan_to_num(
            mixed_state,
            nan=0.0,
            posinf=self.state_clamp_value,
            neginf=-self.state_clamp_value,
        )

        if self.state_clamp_value > 0:
            mixed_state = torch.clamp(mixed_state, -self.state_clamp_value, self.state_clamp_value)

        next_state = torch.tanh(mixed_state)
        next_state = self.dropout(next_state)
        next_state = self.layer_norm(next_state)

        if self.match_input_norm:
            # Keep temporal state magnitude aligned with the current HGNN embedding scale
            # to avoid oversized dot products in downstream BPR loss.
            target_norm = torch.norm(current_state, dim=1, keepdim=True).detach().clamp_min(1e-6)
            next_norm = torch.norm(next_state, dim=1, keepdim=True).clamp_min(1e-6)
            next_state = next_state * (target_norm / next_norm)

        next_state = torch.nan_to_num(
            next_state,
            nan=0.0,
            posinf=self.state_clamp_value,
            neginf=-self.state_clamp_value,
        )

        if self.state_clamp_value > 0:
            next_state = torch.clamp(next_state, -self.state_clamp_value, self.state_clamp_value)

        return next_state


class TemporalHGNN(nn.Module):
    """Temporal wrapper for HGNN models with sequence state updates.

    Inputs:
        graph_sequence: list of graph inputs [G_t1, G_t2, ...].
            Each element can be:
            - tuple/list of positional args for base HGNN forward
            - dict with keys {"args": tuple/list, "kwargs": dict}
            - dict with hypergraph keys used in this project:
              norm_mashup_call_1, norm_mashup_call_2,
              norm_api_call_1, norm_api_call_2,
              norm_mashup_tag_1, norm_mashup_tag_2,
              norm_api_tag_1, norm_api_tag_2
              and optional global_graph=(global_1, global_2) provided to forward.

    Processing:
        1) Apply base HGNN at each time step
        2) Update sequential states using simple SSM block

    Output:
        Base HGNN-style output tuple where the first two embeddings are replaced by temporal final embeddings,
        preserving compatibility with contrastive and multitask losses.
    """

    HYPERGRAPH_KEYS = (
        "norm_mashup_call_1",
        "norm_mashup_call_2",
        "norm_api_call_1",
        "norm_api_call_2",
        "norm_mashup_tag_1",
        "norm_mashup_tag_2",
        "norm_api_tag_1",
        "norm_api_tag_2",
    )

    def __init__(
        self,
        base_hgnn: nn.Module,
        embedding_dim: int,
        grad_clip_norm: float = 5.0,
        state_clamp_value: float = 10.0,
        dropout: float = 0.0,
        detach_state_between_steps: bool = False,
        debug: bool = False,
        debug_log_every: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.base_hgnn = base_hgnn
        self.embedding_dim = int(embedding_dim)
        self.grad_clip_norm = float(grad_clip_norm)
        self.state_clamp_value = float(state_clamp_value)
        self.detach_state_between_steps = bool(detach_state_between_steps)
        self.debug = bool(debug)
        self.debug_log_every = max(1, int(debug_log_every))

        self.mashup_ssm = SimpleSSMBlock(
            embedding_dim=self.embedding_dim,
            state_clamp_value=self.state_clamp_value,
            dropout=dropout,
        )
        self.api_ssm = SimpleSSMBlock(
            embedding_dim=self.embedding_dim,
            state_clamp_value=self.state_clamp_value,
            dropout=dropout,
        )

        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.latest_norm_stats: List[Dict[str, float]] = []

    def _graph_step_to_call(
        self,
        graph_step: GraphStep,
        global_graph: Optional[Tuple[Any, Any]] = None,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if isinstance(graph_step, (list, tuple)):
            return tuple(graph_step), {}

        if isinstance(graph_step, dict):
            if "args" in graph_step or "kwargs" in graph_step:
                args = tuple(graph_step.get("args", ()))
                kwargs = dict(graph_step.get("kwargs", {}))
                return args, kwargs

            if all(key in graph_step for key in self.HYPERGRAPH_KEYS):
                args = [graph_step[key] for key in self.HYPERGRAPH_KEYS]
                if global_graph is not None:
                    if not isinstance(global_graph, (tuple, list)) or len(global_graph) != 2:
                        raise ValueError("global_graph must be a tuple/list (global_1, global_2)")
                    args.extend([global_graph[0], global_graph[1]])
                return tuple(args), {}

            return tuple(), dict(graph_step)

        raise TypeError(
            f"Unsupported graph_step type: {type(graph_step)}. Use tuple/list or dict."
        )

    def _extract_embeddings(self, model_output: Union[Tuple[Any, ...], List[Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(model_output, (tuple, list)):
            raise TypeError("Base HGNN output must be tuple/list for TemporalHGNN compatibility")
        if len(model_output) < 2:
            raise ValueError("Base HGNN output must have at least two tensors: mashup_emb and api_emb")

        mashup_emb = model_output[0]
        api_emb = model_output[1]

        if not torch.is_tensor(mashup_emb) or not torch.is_tensor(api_emb):
            raise TypeError("First two outputs of base HGNN must be torch.Tensor")

        if mashup_emb.dim() != 2 or api_emb.dim() != 2:
            raise ValueError(
                f"Expected 2D embeddings, got mashup={tuple(mashup_emb.shape)}, api={tuple(api_emb.shape)}"
            )

        if mashup_emb.size(1) != self.embedding_dim or api_emb.size(1) != self.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch in TemporalHGNN: "
                f"mashup={mashup_emb.size(1)}, api={api_emb.size(1)}, expected={self.embedding_dim}"
            )

        return mashup_emb, api_emb

    def _log_norms(self, step_id: int, total_steps: int, mashup_state: torch.Tensor, api_state: torch.Tensor) -> None:
        mashup_norm = float(torch.norm(mashup_state, dim=1).mean().detach().cpu())
        api_norm = float(torch.norm(api_state, dim=1).mean().detach().cpu())
        max_abs = float(
            torch.max(
                torch.stack(
                    [
                        mashup_state.detach().abs().max(),
                        api_state.detach().abs().max(),
                    ]
                )
            ).cpu()
        )

        stat = {
            "step": float(step_id),
            "mashup_norm_mean": mashup_norm,
            "api_norm_mean": api_norm,
            "max_abs_value": max_abs,
        }
        self.latest_norm_stats.append(stat)

        if self.debug and ((step_id + 1) % self.debug_log_every == 0 or step_id + 1 == total_steps):
            self.logger.info(
                "[TemporalHGNN] t=%d/%d mashup_norm=%.6f api_norm=%.6f max_abs=%.6f",
                step_id + 1,
                total_steps,
                mashup_norm,
                api_norm,
                max_abs,
            )

    def _replace_output_embeddings(
        self,
        base_output: Union[Tuple[Any, ...], List[Any]],
        final_mashup: torch.Tensor,
        final_api: torch.Tensor,
    ) -> Tuple[Any, ...]:
        out = list(base_output)
        out[0] = final_mashup
        out[1] = final_api

        # Keep compatibility with multitask heads by refreshing logits when predictor heads exist.
        if hasattr(self.base_hgnn, "mashup_tag_predictor") and hasattr(self.base_hgnn, "api_tag_predictor"):
            try:
                mashup_logits = self.base_hgnn.mashup_tag_predictor(final_mashup)
                api_logits = self.base_hgnn.api_tag_predictor(final_api)

                if len(out) >= 4 and torch.is_tensor(out[-2]) and torch.is_tensor(out[-1]):
                    if out[-2].shape == mashup_logits.shape and out[-1].shape == api_logits.shape:
                        out[-2] = mashup_logits
                        out[-1] = api_logits
                elif len(out) == 2:
                    out.extend([mashup_logits, api_logits])
            except Exception:
                pass

        # Keep compatibility with contrastive losses by refreshing local projections when present.
        if hasattr(self.base_hgnn, "mashup_local_proj") and hasattr(self.base_hgnn, "api_local_proj"):
            try:
                if len(out) >= 8 and torch.is_tensor(out[6]) and torch.is_tensor(out[7]):
                    out[6] = self.base_hgnn.mashup_local_proj(final_mashup)
                    out[7] = self.base_hgnn.api_local_proj(final_api)
            except Exception:
                pass

        return tuple(out)

    def forward(
        self,
        graph_sequence: Sequence[GraphStep],
        global_graph: Optional[Tuple[TensorOrAny, TensorOrAny]] = None,
        return_all_states: bool = False,
    ) -> Union[
        Tuple[Any, ...],
        Tuple[Tuple[Any, ...], List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, float]]],
    ]:
        if not isinstance(graph_sequence, (list, tuple)) or len(graph_sequence) == 0:
            raise ValueError("graph_sequence must be a non-empty list/tuple of time-step graph inputs")

        temporal_mashup_state: Optional[torch.Tensor] = None
        temporal_api_state: Optional[torch.Tensor] = None
        base_output_last: Optional[Union[Tuple[Any, ...], List[Any]]] = None
        state_history: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self.latest_norm_stats = []

        for step_idx, graph_step in enumerate(graph_sequence):
            step_args, step_kwargs = self._graph_step_to_call(graph_step, global_graph=global_graph)

            base_output = self.base_hgnn(*step_args, **step_kwargs)
            current_mashup, current_api = self._extract_embeddings(base_output)

            if temporal_mashup_state is None:
                temporal_mashup_state = current_mashup
                temporal_api_state = current_api
            else:
                if temporal_mashup_state.shape != current_mashup.shape:
                    raise ValueError(
                        "Temporal mashup state shape mismatch: "
                        f"prev={tuple(temporal_mashup_state.shape)} current={tuple(current_mashup.shape)}"
                    )
                if temporal_api_state.shape != current_api.shape:
                    raise ValueError(
                        "Temporal api state shape mismatch: "
                        f"prev={tuple(temporal_api_state.shape)} current={tuple(current_api.shape)}"
                    )

                prev_mashup = temporal_mashup_state.detach() if self.detach_state_between_steps else temporal_mashup_state
                prev_api = temporal_api_state.detach() if self.detach_state_between_steps else temporal_api_state

                temporal_mashup_state = self.mashup_ssm(prev_mashup, current_mashup)
                temporal_api_state = self.api_ssm(prev_api, current_api)

            temporal_mashup_state = torch.nan_to_num(
                temporal_mashup_state,
                nan=0.0,
                posinf=self.state_clamp_value,
                neginf=-self.state_clamp_value,
            )
            temporal_api_state = torch.nan_to_num(
                temporal_api_state,
                nan=0.0,
                posinf=self.state_clamp_value,
                neginf=-self.state_clamp_value,
            )

            if self.state_clamp_value > 0:
                temporal_mashup_state = torch.clamp(
                    temporal_mashup_state,
                    -self.state_clamp_value,
                    self.state_clamp_value,
                )
                temporal_api_state = torch.clamp(
                    temporal_api_state,
                    -self.state_clamp_value,
                    self.state_clamp_value,
                )

            if not torch.isfinite(temporal_mashup_state).all() or not torch.isfinite(temporal_api_state).all():
                raise RuntimeError(
                    f"Non-finite temporal states detected at step={step_idx}. "
                    "Try lower learning rate or stronger clipping."
                )

            self._log_norms(step_idx, len(graph_sequence), temporal_mashup_state, temporal_api_state)

            if return_all_states:
                state_history.append((temporal_mashup_state, temporal_api_state))

            base_output_last = base_output

        if base_output_last is None:
            raise RuntimeError("TemporalHGNN failed to compute outputs from graph sequence")

        final_output = self._replace_output_embeddings(
            base_output_last,
            temporal_mashup_state,
            temporal_api_state,
        )

        if return_all_states:
            return final_output, state_history, self.latest_norm_stats
        return final_output

    def clip_gradients(self, max_norm: Optional[float] = None) -> torch.Tensor:
        """Apply gradient clipping to reduce exploding gradients."""
        clip_value = self.grad_clip_norm if max_norm is None else float(max_norm)
        if clip_value <= 0:
            return torch.tensor(0.0)

        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return torch.tensor(0.0)

        return torch.nn.utils.clip_grad_norm_(params, max_norm=clip_value)

    def gradient_health_check(self) -> Tuple[bool, str]:
        """Check for NaN/Inf gradients and return (is_healthy, first_bad_parameter_name)."""
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False, name
        return True, ""


def build_temporal_graph_sequence(
    temporal_sub_hypergraphs: Sequence[Dict[str, Any]],
    global_1: Optional[Any] = None,
    global_2: Optional[Any] = None,
) -> List[Tuple[Any, ...]]:
    """Utility: convert sub-hypergraph dicts into ordered HGNN call args.

    Each returned tuple is compatible with HGNN forward signature used in this project.
    """
    sequence: List[Tuple[Any, ...]] = []

    for sub_graph in temporal_sub_hypergraphs:
        args = (
            sub_graph["norm_mashup_call_1"],
            sub_graph["norm_mashup_call_2"],
            sub_graph["norm_api_call_1"],
            sub_graph["norm_api_call_2"],
            sub_graph["norm_mashup_tag_1"],
            sub_graph["norm_mashup_tag_2"],
            sub_graph["norm_api_tag_1"],
            sub_graph["norm_api_tag_2"],
        )
        if global_1 is not None and global_2 is not None:
            args = args + (global_1, global_2)
        sequence.append(args)

    return sequence


def usage_example(
    base_hgnn: nn.Module,
    temporal_graph_sequence: Sequence[GraphStep],
    embedding_dim: int,
    global_graph: Optional[Tuple[Any, Any]] = None,
) -> Tuple[Any, ...]:
    """Minimal usage example for training loops.

    Typical training step:
        temporal_model = TemporalHGNN(base_hgnn=model, embedding_dim=768, debug=True)
        outputs = temporal_model(temporal_graph_sequence, global_graph=(global_1, global_2))
        loss = ...
        loss.backward()
        temporal_model.clip_gradients(5.0)
        optimizer.step()
    """
    temporal_model = TemporalHGNN(
        base_hgnn=base_hgnn,
        embedding_dim=embedding_dim,
        grad_clip_norm=5.0,
        state_clamp_value=10.0,
        debug=True,
        debug_log_every=1,
    )

    outputs = temporal_model(
        graph_sequence=temporal_graph_sequence,
        global_graph=global_graph,
    )

    return outputs
