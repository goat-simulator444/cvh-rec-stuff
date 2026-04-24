import torch.optim as optim
import sys
import math
import os
from Models import HCF
from utility.helper import ensureDir
from utility.batch_test import data_generator
import h5py
from scipy.spatial.distance import cosine
import numpy as np
import torch
import warnings
import random
import torch.nn.functional as F
import scipy.sparse as sp
from time import time
import torch.sparse as sparse
from utility.parser import parse_args
from utility.text_model_utils import (
    embedding_dir,
    resolve_embed_size,
    resolve_embedding_file,
    text_model_tag,
)
from utility.load_data import TemporalDataset
from models.diffusion import EdgeDiffusion, generate_synthetic_edges
from models.temporal_hgnn import TemporalHGNN, build_temporal_graph_sequence
args = parse_args()

warnings.filterwarnings("ignore")



def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)


result = []
alpha1 = args.alpha1
alpha2 = args.alpha2
alpha3 = args.alpha3
alpha4 = args.alpha4

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
try:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except Exception:
    pass


def resolve_runtime_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            return torch.device('cuda')
        except Exception as e:
            print(f"CUDA unavailable, fallback to CPU: {e}")
    return torch.device('cpu')


def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)  # 获取对角线上的元素
    union = square_sum[:, None] + square_sum - intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.divide(intersection, union)
    return np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)


class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data
        self.n_mashup = data_config['n_users']
        self.n_api = data_config['n_items']
        self.n_tag = data_config['n_tags']

        self.record_alphas = False
        self.device = resolve_runtime_device()
        self.lr = args.lr
        self.use_bge = args.use_bge == 1
        self.emb_dim = resolve_embed_size(args.embed_size, args.text_model)
        if self.emb_dim != args.embed_size:
            print(f"Adjusted embed_size from {args.embed_size} to {self.emb_dim} for text model {args.text_model}")
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.model_type += '_%s_%s_layers%d' % (self.adj_type, self.alg_type, self.layer_num)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        method = 'CVH-REC'
        model_tag = text_model_tag(args.text_model)
        self.embedding_dir = embedding_dir(args.embedding_base_dir, args.text_model)
        if not os.path.exists(self.embedding_dir):
            self.embedding_dir = args.embedding_base_dir

        alpha_str = f"{args.alpha1}-{args.alpha2}-{args.alpha3}-{args.alpha4}"
        reg_str = '-'.join(str(r) for r in eval(args.regs))

        self.weights_save_path = (
            f"{args.weights_path}weights/{method}/{model_tag}/{args.dataset}/{args.seed}/"
            f"alpha-{alpha_str}/"
            f"layer_num{args.layer_num}/"
            f"mess_dropout{args.mess_dropout}/"
            f"drop_edge{args.drop_edge}/"
            f"lr{args.lr}/"
            f"reg{reg_str}"
        )

        str(args.similar_threthold) + str(args.alpha1) + str(args.alpha2) + str(args.alpha3) + str(args.alpha4)
        self.result_message = []

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['hcf']:
            self.model = HCF(
                self.n_mashup,
                self.n_api,
                self.n_tag,
                self.emb_dim,
                self.layer_num,
                self.mess_dropout,
                embedding_dir=self.embedding_dir,
            )
        else:
            raise Exception('Dont know which model to train')

        self.use_temporal = args.use_temporal == 1
        self.temporal_model = None
        self.temporal_graph_sequence = []
        self.temporal_call_edge_counts = []
        self.last_temporal_step_count = 0
        self.runtime_temporal_max_steps = max(0, int(args.temporal_max_steps))
        self.temporal_oom_backoff_count = 0
        self.temporal_time_slices = []
        self.temporal_sub_hypergraphs = []

        self.use_diffusion = args.use_diffusion == 1
        if args.use_diffusion == 1 and args.edge_diffusion == 0:
            print('[Config][WARN] --edge_diffusion=0 is overridden by --use_diffusion=1 (ablation switch has priority).')
        elif args.use_diffusion == 0 and args.edge_diffusion == 1:
            print('[Config] Diffusion disabled by --use_diffusion=0.')
        self.use_edge_diffusion = self.use_diffusion
        self.edge_diffusion = None
        self.edge_diffusion_weight = max(0.0, float(args.edge_diffusion_weight))
        self.edge_generation_threshold = float(args.edge_gen_threshold)
        self.edge_refresh_interval = max(0, int(args.edge_gen_refresh_interval))
        self.last_generated_edge_count = 0
        self.real_edge_index = self._build_real_edge_index()
        self.last_augmented_edge_count = int(self.real_edge_index.size(1))

        if self.use_edge_diffusion:
            self.edge_diffusion = EdgeDiffusion(
                embedding_dim=self.emb_dim,
                hidden_dim=args.edge_diffusion_hidden_dim,
                diffusion_steps=args.edge_diffusion_steps,
            )

        dataset_dir = args.data_path + args.dataset
        time_sliced_call_graphs = self._load_temporal_call_graphs(dataset_dir)

        (self.norm_mashup_call_1, self.norm_mashup_call_2, self.norm_api_call_1, self.norm_api_call_2,
         self.norm_mashup_tag_1, self.norm_mashup_tag_2, self.norm_api_tag_1,
         self.norm_api_tag_2) = self.build_temporal_hypergraph(
            dataset_dir + '/TE.csv',
            dataset_dir + '/mashup_tags.csv',
            dataset_dir + '/api_tags.csv',
            time_sliced_data=time_sliced_call_graphs,
        )
        self.last_augmented_edge_count = int(getattr(self, 'main_call_edge_count', self.last_augmented_edge_count))
        self.global_1, self.global_2 = self.build_global_hyperedge(
            dataset_dir + '/TE.csv',
            dataset_dir + '/mashup_tags.csv',
            dataset_dir + '/api_tags.csv',
            self.n_mashup,
            self.n_api,
            self.n_tag,
        )

        self.mashup_tag_labels = self.load_tag_labels(
            args.data_path + args.dataset + '/mashup_tags.csv',
            self.n_mashup, self.n_tag
        )
        self.api_tag_labels = self.load_tag_labels(
            args.data_path + args.dataset + '/api_tags.csv',
            self.n_api, self.n_tag
        )
        self.tag_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.model = self.model.to(self.device)
        if self.edge_diffusion is not None:
            self.edge_diffusion = self.edge_diffusion.to(self.device)

        self.real_edge_index = self.real_edge_index.to(self.device)
        self.norm_mashup_call_1 = self.norm_mashup_call_1.to(self.device)
        self.norm_mashup_call_2 = self.norm_mashup_call_2.to(self.device)
        self.norm_api_call_1 = self.norm_api_call_1.to(self.device)
        self.norm_api_call_2 = self.norm_api_call_2.to(self.device)
        self.norm_mashup_tag_1 = self.norm_mashup_tag_1.to(self.device)
        self.norm_mashup_tag_2 = self.norm_mashup_tag_2.to(self.device)
        self.norm_api_tag_1 = self.norm_api_tag_1.to(self.device)
        self.norm_api_tag_2 = self.norm_api_tag_2.to(self.device)
        self.global_1 = self.global_1.to(self.device)
        self.global_2 = self.global_2.to(self.device)
        self.mashup_tag_labels = self.mashup_tag_labels.to(self.device)
        self.api_tag_labels = self.api_tag_labels.to(self.device)

        self._move_temporal_sub_hypergraphs_to_device()
        self._refresh_temporal_graph_sequence()

        if self.use_temporal:
            self.temporal_model = TemporalHGNN(
                base_hgnn=self.model,
                embedding_dim=self.emb_dim,
                grad_clip_norm=5.0,
                state_clamp_value=10.0,
                detach_state_between_steps=False,
                debug=args.temporal_debug_logs == 1,
                debug_log_every=max(1, int(args.temporal_debug_log_every)),
            )
            self.temporal_model = self.temporal_model.to(self.device)

            self.last_temporal_step_count = len(self.temporal_graph_sequence)
            if self.last_temporal_step_count <= 1:
                print('[TemporalHGNN][WARN] Temporal mode enabled but only one step available; behavior is close to static HGNN.')

        if self.temporal_model is not None:
            opt_params = list(self.temporal_model.parameters())
        else:
            opt_params = list(self.model.parameters())

        if self.edge_diffusion is not None:
            opt_params += list(self.edge_diffusion.parameters())

        self.optimizer = optim.Adam(opt_params, lr=self.lr)

        self.active_modules = {
            'bge': self.use_bge,
            'temporal': self.use_temporal,
            'diffusion': self.use_diffusion,
        }
        print(
            f"[ModuleStatus] active={self.active_modules}, text_model={args.text_model}, "
            f"embedding_dir={self.embedding_dir}"
        )

        print(
            f"[TrainConfig] use_bge={self.use_bge}, use_temporal={self.use_temporal}, "
            f"temporal_steps={len(self.temporal_graph_sequence)}, use_diffusion={self.use_diffusion}, "
            f"real_edges={int(self.real_edge_index.size(1))}, "
            f"runtime_temporal_max_steps={self.runtime_temporal_max_steps}"
        )

        self.lr_scheduler = self.set_lr_scheduler()

    def get_D_inv(self, Hadj):

        H = sp.coo_matrix(Hadj.shape)
        H.row = Hadj.row.copy()
        H.col = Hadj.col.copy()
        H.data = Hadj.data.copy()
        rowsum = np.array(H.sum(1))
        columnsum = np.array(H.sum(0))

        Dv_inv = np.power(rowsum, -1).flatten()
        De_inv = np.power(columnsum, -1).flatten()
        Dv_inv[np.isinf(Dv_inv)] = 0.
        De_inv[np.isinf(De_inv)] = 0.

        Dv_mat_inv = sp.diags(Dv_inv)
        De_mat_inv = sp.diags(De_inv)
        return Dv_mat_inv, De_mat_inv

    def _build_real_edge_index(self):
        users = []
        items = []
        for uid, item_list in data_generator.train_items.items():
            if uid < 0 or uid >= self.n_mashup:
                continue
            for iid in item_list:
                if 0 <= iid < self.n_api:
                    users.append(uid)
                    items.append(iid)

        if not users:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor([users, items], dtype=torch.long)
        flat = edge_index[0] * self.n_api + edge_index[1]
        unique_flat = torch.unique(flat)

        dedup_users = torch.div(unique_flat, self.n_api, rounding_mode='floor')
        dedup_items = unique_flat % self.n_api
        return torch.stack([dedup_users, dedup_items], dim=0)

    def _rebuild_main_hypergraph_from_call_matrix(self, mashup_api_call_matrix):
        if mashup_api_call_matrix.shape != (self.n_mashup, self.n_api):
            raise AssertionError(
                f"Augmented call matrix shape mismatch: {mashup_api_call_matrix.shape} != {(self.n_mashup, self.n_api)}"
            )

        if not np.isfinite(mashup_api_call_matrix).all():
            raise AssertionError("Augmented call matrix contains NaN/Inf")

        if not hasattr(self, 'cached_mashup_tag_call_matrix') or not hasattr(self, 'cached_api_tag_call_matrix'):
            raise RuntimeError("Tag relation matrices are not initialized for graph augmentation")

        main_hypergraph = self._build_hypergraph_bundle(
            mashup_api_call_matrix.astype(np.float32, copy=False),
            self.cached_mashup_tag_call_matrix,
            self.cached_api_tag_call_matrix,
            edge_suffix="",
            store_attr=True,
        )

        (self.norm_mashup_call_1, self.norm_mashup_call_2,
         self.norm_api_call_1, self.norm_api_call_2,
         self.norm_mashup_tag_1, self.norm_mashup_tag_2,
         self.norm_api_tag_1, self.norm_api_tag_2) = main_hypergraph

        self.norm_mashup_call_1 = self.norm_mashup_call_1.to(self.device)
        self.norm_mashup_call_2 = self.norm_mashup_call_2.to(self.device)
        self.norm_api_call_1 = self.norm_api_call_1.to(self.device)
        self.norm_api_call_2 = self.norm_api_call_2.to(self.device)
        self.norm_mashup_tag_1 = self.norm_mashup_tag_1.to(self.device)
        self.norm_mashup_tag_2 = self.norm_mashup_tag_2.to(self.device)
        self.norm_api_tag_1 = self.norm_api_tag_1.to(self.device)
        self.norm_api_tag_2 = self.norm_api_tag_2.to(self.device)

        self.main_hypergraph = {
            'norm_mashup_call_1': self.norm_mashup_call_1,
            'norm_mashup_call_2': self.norm_mashup_call_2,
            'norm_api_call_1': self.norm_api_call_1,
            'norm_api_call_2': self.norm_api_call_2,
            'norm_mashup_tag_1': self.norm_mashup_tag_1,
            'norm_mashup_tag_2': self.norm_mashup_tag_2,
            'norm_api_tag_1': self.norm_api_tag_1,
            'norm_api_tag_2': self.norm_api_tag_2,
        }
        self.current_call_matrix = mashup_api_call_matrix
        self.main_call_edge_count = int(np.count_nonzero(mashup_api_call_matrix))

    def _move_temporal_sub_hypergraphs_to_device(self):
        if not hasattr(self, 'temporal_sub_hypergraphs'):
            self.temporal_sub_hypergraphs = []
            return

        tensor_keys = (
            'norm_mashup_call_1', 'norm_mashup_call_2',
            'norm_api_call_1', 'norm_api_call_2',
            'norm_mashup_tag_1', 'norm_mashup_tag_2',
            'norm_api_tag_1', 'norm_api_tag_2',
        )

        # Keep cached temporal sub-graphs on CPU to avoid high peak GPU memory.
        # Selected temporal steps are moved to runtime device lazily in
        # _refresh_temporal_graph_sequence after stride/max-step filtering.
        moved_to_cpu = 0
        for sub_graph in self.temporal_sub_hypergraphs:
            for key in tensor_keys:
                if key in sub_graph and torch.is_tensor(sub_graph[key]):
                    if sub_graph[key].device.type != 'cpu':
                        sub_graph[key] = sub_graph[key].cpu()
                        moved_to_cpu += 1

        if len(self.temporal_sub_hypergraphs) > 0:
            print(
                f"[TemporalHypergraph] Keeping {len(self.temporal_sub_hypergraphs)} sub-hypergraphs on CPU "
                f"(moved_to_cpu={moved_to_cpu}) and moving selected steps lazily to {self.device}."
            )

    def _prepare_runtime_temporal_sub_graph(self, sub_graph):
        tensor_keys = (
            'norm_mashup_call_1', 'norm_mashup_call_2',
            'norm_api_call_1', 'norm_api_call_2',
            'norm_mashup_tag_1', 'norm_mashup_tag_2',
            'norm_api_tag_1', 'norm_api_tag_2',
        )

        runtime_graph = {}
        for key, value in sub_graph.items():
            if key in tensor_keys and torch.is_tensor(value):
                runtime_graph[key] = value.to(self.device)
            else:
                runtime_graph[key] = value
        return runtime_graph

    def _refresh_temporal_graph_sequence(self):
        main_step = (
            self.norm_mashup_call_1, self.norm_mashup_call_2,
            self.norm_api_call_1, self.norm_api_call_2,
            self.norm_mashup_tag_1, self.norm_mashup_tag_2,
            self.norm_api_tag_1, self.norm_api_tag_2,
            self.global_1, self.global_2,
        )

        sequence = []
        edge_counts = []

        if self.use_temporal and len(self.temporal_sub_hypergraphs) > 0:
            selected_sub_graphs = list(self.temporal_sub_hypergraphs)
            requested_steps = len(selected_sub_graphs)
            selected_for_runtime = list(selected_sub_graphs)
            runtime_sub_graphs = []
            while True:
                try:
                    runtime_sub_graphs = [
                        self._prepare_runtime_temporal_sub_graph(sg)
                        for sg in selected_for_runtime
                    ]
                    break
                except torch.OutOfMemoryError:
                    runtime_sub_graphs = []
                    if self.device.type != 'cuda' or len(selected_for_runtime) <= 1:
                        raise

                    torch.cuda.empty_cache()
                    reduced_steps = max(1, len(selected_for_runtime) // 2)
                    if reduced_steps >= len(selected_for_runtime):
                        reduced_steps = len(selected_for_runtime) - 1

                    print(
                        f"[TemporalHypergraph][WARN] CUDA OOM while moving temporal steps to GPU. "
                        f"Reducing steps from {len(selected_for_runtime)} to {reduced_steps}."
                    )
                    selected_for_runtime = selected_for_runtime[-reduced_steps:]

            if len(selected_for_runtime) != requested_steps:
                print(
                    f"[TemporalHypergraph] Using {len(selected_for_runtime)} temporal steps at runtime "
                    f"(requested {requested_steps} built temporal steps)."
                )

            selected_sub_graphs = selected_for_runtime

            # Each temporal HGNN step needs global hypergraph tensors to satisfy HCF.forward signature.
            sequence.extend(
                build_temporal_graph_sequence(
                    runtime_sub_graphs,
                    self.global_1,
                    self.global_2,
                )
            )
            edge_counts.extend([int(sub.get('call_edge_count', 0)) for sub in selected_sub_graphs])

        sequence.append(main_step)
        edge_counts.append(int(getattr(self, 'main_call_edge_count', 0)))

        self.temporal_graph_sequence = sequence
        self.temporal_call_edge_counts = edge_counts
        self.last_temporal_step_count = len(sequence)

    def _reduce_temporal_memory_after_oom(self, stage):
        recovered = False

        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.temporal_model is not None and not self.temporal_model.detach_state_between_steps:
            self.temporal_model.detach_state_between_steps = True
            recovered = True
            print(
                f"[TemporalHGNN][OOM] {stage}: enabled detach_state_between_steps=True to reduce activation memory."
            )

        if self.use_temporal and len(self.temporal_time_slices) > 0:
            stride = max(1, int(args.temporal_stride))
            available_steps = len(list(range(0, len(self.temporal_time_slices), stride)))
            current_limit = self.runtime_temporal_max_steps if self.runtime_temporal_max_steps > 0 else available_steps

            if current_limit > 1:
                reduced_limit = max(1, current_limit // 2)
                if reduced_limit < current_limit:
                    self.runtime_temporal_max_steps = reduced_limit
                    self._rebuild_temporal_sub_hypergraphs()
                    self._refresh_temporal_graph_sequence()
                    recovered = True
                    print(
                        f"[TemporalHGNN][OOM] {stage}: reduced runtime_temporal_max_steps "
                        f"from {current_limit} to {reduced_limit}; current temporal_steps={self.last_temporal_step_count}."
                    )

        if recovered:
            self.temporal_oom_backoff_count += 1

        return recovered

    @staticmethod
    def _is_cuda_oom_like_error(error):
        if isinstance(error, torch.OutOfMemoryError):
            return True

        if not isinstance(error, RuntimeError):
            return False

        message = str(error).lower()
        oom_markers = (
            'out of memory',
            'cublas_status_alloc_failed',
            'cuda malloc',
        )
        return any(marker in message for marker in oom_markers)

    def _forward_hgnn(self):
        if self.use_temporal and self.temporal_model is not None:
            return self.temporal_model(
                graph_sequence=self.temporal_graph_sequence,
                global_graph=(self.global_1, self.global_2),
                return_all_states=False,
            )

        return self.model(
            self.norm_mashup_call_1, self.norm_mashup_call_2,
            self.norm_api_call_1, self.norm_api_call_2,
            self.norm_mashup_tag_1, self.norm_mashup_tag_2,
            self.norm_api_tag_1, self.norm_api_tag_2,
            self.global_1, self.global_2,
        )

    def _refresh_augmented_hypergraph(self, epoch_id):
        if self.edge_diffusion is None:
            return

        if self.real_edge_index.numel() == 0:
            print('[EdgeDiffusion][WARN] No real edges found; skip graph augmentation.')
            return

        self.model.eval()
        if self.temporal_model is not None:
            self.temporal_model.eval()
        self.edge_diffusion.eval()

        real_edge_count = int(self.real_edge_index.size(1))
        configured_max_total_edges = args.edge_gen_max_total_edges if args.edge_gen_max_total_edges > 0 else None
        max_ratio_to_real = float(args.edge_gen_max_ratio_to_real)
        dynamic_max_total_edges = configured_max_total_edges
        if max_ratio_to_real > 0.0 and real_edge_count > 0:
            ratio_cap_edges = max(1, int(real_edge_count * max_ratio_to_real))
            if dynamic_max_total_edges is None:
                dynamic_max_total_edges = ratio_cap_edges
            else:
                dynamic_max_total_edges = min(dynamic_max_total_edges, ratio_cap_edges)

        if dynamic_max_total_edges is not None and dynamic_max_total_edges != configured_max_total_edges:
            print(
                f"[EdgeDiffusion] Applying ratio cap: max_total_edges={dynamic_max_total_edges} "
                f"(configured={configured_max_total_edges}, real_edges={real_edge_count}, "
                f"max_ratio_to_real={max_ratio_to_real:.3f})"
            )

        with torch.no_grad():
            outputs = self._forward_hgnn()
            m_embeddings, a_embeddings = outputs[0], outputs[1]

            generated_edge_index, generated_scores, gen_stats = generate_synthetic_edges(
                edge_generator=self.edge_diffusion,
                mashup_embeddings=m_embeddings,
                api_embeddings=a_embeddings,
                threshold=self.edge_generation_threshold,
                real_edge_index=self.real_edge_index,
                max_edges_per_node=max(0, int(args.edge_gen_max_edges_per_node)),
                topk_per_node=args.edge_gen_topk_per_node if args.edge_gen_topk_per_node > 0 else None,
                max_total_edges=dynamic_max_total_edges,
                threshold_step=max(0.0, float(args.edge_gen_threshold_step)),
                max_threshold_tuning_steps=max(0, int(args.edge_gen_threshold_tuning_steps)),
                chunk_size=max(256, int(args.edge_gen_chunk_size)),
            )

        if generated_edge_index.numel() == 0 and self.last_augmented_edge_count == real_edge_count:
            self.last_generated_edge_count = 0
            self.edge_generation_threshold = float(gen_stats.get('effective_threshold', self.edge_generation_threshold))
            print(
                f"[EdgeDiffusion] epoch={epoch_id} generated=0; skip hypergraph rebuild (already using real graph). "
                f"threshold={self.edge_generation_threshold:.3f}"
            )
            return

        augmented_edge_index = self.edge_diffusion.augment_real_graph(
            real_edge_index=self.real_edge_index,
            generated_edge_index=generated_edge_index,
            num_items=self.n_api,
        )

        duplicates_removed = (
            int(self.real_edge_index.size(1))
            + int(generated_edge_index.size(1))
            - int(augmented_edge_index.size(1))
        )

        augmented_call_matrix = self.edge_diffusion.edge_index_to_dense_matrix(
            edge_index=augmented_edge_index,
            num_users=self.n_mashup,
            num_items=self.n_api,
            dtype=torch.float32,
        ).detach().cpu().numpy().astype(np.float32, copy=False)

        self._rebuild_main_hypergraph_from_call_matrix(augmented_call_matrix)
        self._refresh_temporal_graph_sequence()

        self.last_generated_edge_count = int(generated_edge_index.size(1))
        self.last_augmented_edge_count = int(augmented_edge_index.size(1))
        self.edge_generation_threshold = float(gen_stats.get('effective_threshold', self.edge_generation_threshold))

        print(
            f"[EdgeDiffusion] epoch={epoch_id} real={int(self.real_edge_index.size(1))} "
            f"generated={int(generated_edge_index.size(1))} augmented={int(augmented_edge_index.size(1))} "
            f"duplicates_removed={duplicates_removed} threshold={self.edge_generation_threshold:.3f}"
        )

        if generated_scores.numel() > 0:
            avg_prob = float(generated_scores.mean().item())
            max_prob = float(generated_scores.max().item())
            print(
                f"[EdgeDiffusion] generated edge confidence: mean={avg_prob:.4f}, max={max_prob:.4f}"
            )

    def _read_relation_matrix(self, filename, shape, transpose=False):
        matrix = np.zeros(shape, dtype=np.float32)
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = [item for item in line.split(" ") if item != '']
                if not parts:
                    continue

                try:
                    main_id = int(parts[0])
                except ValueError:
                    continue

                if transpose:
                    if not (0 <= main_id < shape[1]):
                        continue
                else:
                    if not (0 <= main_id < shape[0]):
                        continue

                related_ids = []
                related_limit = shape[0] if transpose else shape[1]
                for token in parts[1:]:
                    try:
                        rid = int(token)
                    except ValueError:
                        continue
                    if 0 <= rid < related_limit:
                        related_ids.append(rid)

                if not related_ids:
                    continue

                if transpose:
                    matrix[related_ids, main_id] = 1.0
                else:
                    matrix[main_id, related_ids] = 1.0

        return matrix

    def _load_temporal_call_graphs(self, dataset_dir):
        if not self.use_temporal:
            self.temporal_sub_hypergraphs = []
            if args.temporal_hypergraph == 1:
                print('[TemporalHypergraph] Skipped because --use_temporal=0.')
            return []

        if args.temporal_hypergraph != 1:
            self.temporal_sub_hypergraphs = []
            return []

        try:
            temporal_dataset = TemporalDataset(
                path=dataset_dir,
                batch_size=self.batch_size,
                interaction_file=args.temporal_interaction_file,
                timestamp_fallback=args.temporal_timestamp_fallback,
            )
            time_slices = temporal_dataset.get_time_slices(args.temporal_window_size)
        except Exception as e:
            print(f"[TemporalHypergraph][WARN] Temporal graph construction disabled due to error: {e}")
            self.temporal_sub_hypergraphs = []
            return []

        if not time_slices:
            self.temporal_sub_hypergraphs = []
            return []

        active_users = set()
        active_items = set()

        for idx, graph in enumerate(time_slices):
            if graph.shape != (self.n_mashup, self.n_api):
                raise AssertionError(
                    f"Time slice {idx} shape mismatch: {graph.shape} != {(self.n_mashup, self.n_api)}"
                )
            if graph.data.size > 0 and not np.isfinite(graph.data).all():
                raise AssertionError(f"Non-finite values found in temporal slice {idx}")

            rows, cols = graph.nonzero()
            active_users.update(rows.tolist())
            active_items.update(cols.tolist())

        missing_users = self.n_mashup - len(active_users)
        missing_items = self.n_api - len(active_items)
        if missing_users > 0 or missing_items > 0:
            print(
                f"[TemporalHypergraph][WARN] Missing active nodes across all time slices: "
                f"users={missing_users}, apis={missing_items}"
            )

        print(
            f"[TemporalHypergraph] Loaded {len(time_slices)} time slices "
            f"from {temporal_dataset.interaction_file} with window_size={args.temporal_window_size}"
        )
        return time_slices

    def _merge_time_sliced_graphs(self, time_sliced_data):
        merged = sp.csr_matrix((self.n_mashup, self.n_api), dtype=np.float32)
        for graph in time_sliced_data:
            merged = merged + graph
        if merged.nnz > 0:
            merged.data = np.ones_like(merged.data, dtype=np.float32)
        merged_dense = merged.toarray().astype(np.float32, copy=False)
        assert merged_dense.shape == (self.n_mashup, self.n_api), "Merged call matrix shape mismatch"
        assert np.isfinite(merged_dense).all(), "Merged call matrix contains NaN/Inf"
        return merged_dense

    def _select_temporal_slice_indices(self):
        if not self.temporal_time_slices:
            return []

        stride = max(1, int(args.temporal_stride))
        selected_indices = list(range(0, len(self.temporal_time_slices), stride))

        min_edges = max(0, int(args.temporal_min_edges_per_slice))
        if min_edges > 0:
            filtered_indices = [
                idx for idx in selected_indices
                if int(self.temporal_time_slices[idx].nnz) >= min_edges
            ]
            if filtered_indices:
                selected_indices = filtered_indices
            else:
                print(
                    f"[TemporalHypergraph][WARN] No temporal slices meet min_edges_per_slice={min_edges}; "
                    "keeping stride-selected slices."
                )

        max_steps = int(self.runtime_temporal_max_steps)
        if max_steps > 0 and len(selected_indices) > max_steps:
            pool_multiplier = max(1, int(args.temporal_recent_pool_multiplier))
            pool_size = max_steps * pool_multiplier
            recent_pool = selected_indices[-pool_size:]

            ranked_pool = sorted(
                recent_pool,
                key=lambda idx: (int(self.temporal_time_slices[idx].nnz), idx),
                reverse=True,
            )
            selected_indices = sorted(ranked_pool[:max_steps])

        return selected_indices

    def _rebuild_temporal_sub_hypergraphs(self):
        self.temporal_sub_hypergraphs = []

        if not self.temporal_time_slices:
            return

        selected_indices = self._select_temporal_slice_indices()
        if not selected_indices:
            print('[TemporalHypergraph][WARN] No temporal slices selected after stride/max-step filtering.')
            return

        built_sub_hypergraphs = []
        for time_step in selected_indices:
            graph = self.temporal_time_slices[time_step]
            call_matrix = graph.toarray().astype(np.float32, copy=False)
            sub_hypergraph = self._build_hypergraph_bundle(
                call_matrix,
                self.cached_mashup_tag_call_matrix,
                self.cached_api_tag_call_matrix,
                edge_suffix=f'_t{time_step}',
                store_attr=False,
            )
            built_sub_hypergraphs.append({
                'time_step': time_step,
                'call_edge_count': int(graph.nnz),
                'norm_mashup_call_1': sub_hypergraph[0],
                'norm_mashup_call_2': sub_hypergraph[1],
                'norm_api_call_1': sub_hypergraph[2],
                'norm_api_call_2': sub_hypergraph[3],
                'norm_mashup_tag_1': sub_hypergraph[4],
                'norm_mashup_tag_2': sub_hypergraph[5],
                'norm_api_tag_1': sub_hypergraph[6],
                'norm_api_tag_2': sub_hypergraph[7],
            })

        self.temporal_sub_hypergraphs = built_sub_hypergraphs
        print(
            f"[TemporalHypergraph] Built {len(self.temporal_sub_hypergraphs)} sub-hypergraphs "
            f"(selected from {len(self.temporal_time_slices)} slices with stride={max(1, int(args.temporal_stride))}, "
            f"runtime_max_steps={self.runtime_temporal_max_steps})."
        )

    def _assert_sparse_tensor_valid(self, tensor, tensor_name):
        values = tensor.coalesce().values()
        if values.numel() == 0:
            return
        if torch.isnan(values).any() or torch.isinf(values).any():
            raise AssertionError(f"{tensor_name} contains NaN/Inf values")

    def _build_and_normalize_hyperedge(self, interaction_matrix, similarity_threshold, node_count, edge_prefix,
                                       store_attr):
        assert interaction_matrix.shape[0] == node_count, (
            f"Interaction matrix row count mismatch for {edge_prefix}: "
            f"{interaction_matrix.shape[0]} != {node_count}"
        )
        if not np.isfinite(interaction_matrix).all():
            raise AssertionError(f"{edge_prefix} interaction matrix contains NaN/Inf")

        jaccard_sim = jaccard_similarity(interaction_matrix)
        if not np.isfinite(jaccard_sim).all():
            raise AssertionError(f"{edge_prefix} jaccard similarity contains NaN/Inf")

        indices = np.where(jaccard_sim >= similarity_threshold)
        values = jaccard_sim[indices].astype(np.float32, copy=False)

        if values.size > 0 and not np.isfinite(values).all():
            raise AssertionError(f"{edge_prefix} hyperedge values contain NaN/Inf")

        hyperedge = sp.coo_matrix((values, indices), shape=(node_count, node_count), dtype=np.float32)
        hyperedge_transpose = hyperedge.T.tocoo()

        Dv, De = self.get_D_inv(hyperedge_transpose)

        Dv_torch = self.sparse_mx_to_torch_sparse_tensor(Dv)
        De_torch = self.sparse_mx_to_torch_sparse_tensor(De)
        hyperedge_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge_transpose)
        hyperedge_transpose_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge)

        temp = sparse.mm(Dv_torch, hyperedge_torch)
        normalized_1 = sparse.mm(temp, De_torch)
        normalized_2 = hyperedge_transpose_torch

        assert normalized_1.shape[0] == node_count and normalized_1.shape[1] == node_count, (
            f"{edge_prefix} normalized_1 shape mismatch: {normalized_1.shape}"
        )
        assert normalized_2.shape[0] == node_count and normalized_2.shape[1] == node_count, (
            f"{edge_prefix} normalized_2 shape mismatch: {normalized_2.shape}"
        )

        self._assert_sparse_tensor_valid(normalized_1, f"{edge_prefix}.normalized_1")
        self._assert_sparse_tensor_valid(normalized_2, f"{edge_prefix}.normalized_2")

        if store_attr:
            setattr(self, f'H{edge_prefix}Edge', hyperedge_torch)
            setattr(self, f'H{edge_prefix}Edge_T', hyperedge_transpose_torch)
            setattr(self, f'norm_{edge_prefix}1', normalized_1)
            setattr(self, f'norm_{edge_prefix}2', normalized_2)

        return normalized_1, normalized_2

    def _build_hypergraph_bundle(self, mashup_api_call_matrix, mashup_tag_call_matrix, api_tag_call_matrix,
                                 edge_suffix="", store_attr=True):
        api_mashup_matrix = mashup_api_call_matrix.T

        norm_mashup_call_1, norm_mashup_call_2 = self._build_and_normalize_hyperedge(
            mashup_api_call_matrix,
            alpha1,
            self.n_mashup,
            f'mashup{edge_suffix}',
            store_attr=store_attr,
        )
        norm_api_call_1, norm_api_call_2 = self._build_and_normalize_hyperedge(
            api_mashup_matrix,
            alpha2,
            self.n_api,
            f'api{edge_suffix}',
            store_attr=store_attr,
        )
        norm_mashup_tag_1, norm_mashup_tag_2 = self._build_and_normalize_hyperedge(
            mashup_tag_call_matrix,
            alpha3,
            self.n_mashup,
            f'mashupTagSim{edge_suffix}',
            store_attr=store_attr,
        )
        norm_api_tag_1, norm_api_tag_2 = self._build_and_normalize_hyperedge(
            api_tag_call_matrix,
            alpha4,
            self.n_api,
            f'apiTagSim{edge_suffix}',
            store_attr=store_attr,
        )

        return (
            norm_mashup_call_1,
            norm_mashup_call_2,
            norm_api_call_1,
            norm_api_call_2,
            norm_mashup_tag_1,
            norm_mashup_tag_2,
            norm_api_tag_1,
            norm_api_tag_2,
        )

    def build_temporal_hypergraph(self, mashup_api_call_file, mashup_tag_call_file, api_tag_call_file,
                                  time_sliced_data=None):
        mashup_tag_call_matrix = self._read_relation_matrix(
            mashup_tag_call_file,
            (self.n_mashup, self.n_tag),
            transpose=False,
        )
        api_tag_call_matrix = self._read_relation_matrix(
            api_tag_call_file,
            (self.n_api, self.n_tag),
            transpose=False,
        )

        self.cached_mashup_tag_call_matrix = mashup_tag_call_matrix.astype(np.float32, copy=False)
        self.cached_api_tag_call_matrix = api_tag_call_matrix.astype(np.float32, copy=False)

        if time_sliced_data:
            mashup_api_call_matrix = self._merge_time_sliced_graphs(time_sliced_data)
        else:
            mashup_api_call_matrix = self._read_relation_matrix(
                mashup_api_call_file,
                (self.n_mashup, self.n_api),
                transpose=False,
            )

        self.current_call_matrix = mashup_api_call_matrix.astype(np.float32, copy=False)
        self.main_call_edge_count = int(np.count_nonzero(mashup_api_call_matrix))

        main_hypergraph = self._build_hypergraph_bundle(
            mashup_api_call_matrix,
            mashup_tag_call_matrix,
            api_tag_call_matrix,
            edge_suffix="",
            store_attr=True,
        )

        self.temporal_time_slices = list(time_sliced_data) if time_sliced_data else []
        self.temporal_sub_hypergraphs = []
        if self.temporal_time_slices:
            self._rebuild_temporal_sub_hypergraphs()

        self.main_hypergraph = {
            'norm_mashup_call_1': main_hypergraph[0],
            'norm_mashup_call_2': main_hypergraph[1],
            'norm_api_call_1': main_hypergraph[2],
            'norm_api_call_2': main_hypergraph[3],
            'norm_mashup_tag_1': main_hypergraph[4],
            'norm_mashup_tag_2': main_hypergraph[5],
            'norm_api_tag_1': main_hypergraph[6],
            'norm_api_tag_2': main_hypergraph[7],
        }

        return main_hypergraph

    def build_hyper_edge(self, mashup_api_call_file, mashup_tag_call_file, api_tag_call_file):
        return self.build_temporal_hypergraph(
            mashup_api_call_file,
            mashup_tag_call_file,
            api_tag_call_file,
            time_sliced_data=None,
        )

    def build_global_hyperedge(self, mashup_api_call_file, mashup_tag_call_file, api_tag_call_file,
                               N_MASHUP, N_API, N_TAG):

        """
        构造global视角的超边：
        - 对每个mashup，构造一个超边：包含mashup自身、它调用的API（0.8）、它包含的tag（0.5）。
        - 对每个API，构造一个超边：包含API自身、它使用的tag（0.5）。
        """

        def read_and_fill_matrix(filename, shape):
            matrix = np.zeros(shape)
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [item for item in line.split(" ") if item != '']
                    main_id = int(parts[0])
                    related_ids = list(map(int, parts[1:]))
                    matrix[main_id, related_ids] = 1
            return matrix

        # 读取交互矩阵
        mashup_api_matrix = read_and_fill_matrix(mashup_api_call_file, (N_MASHUP, N_API))
        mashup_tag_matrix = read_and_fill_matrix(mashup_tag_call_file, (N_MASHUP, N_TAG))
        api_tag_matrix = read_and_fill_matrix(api_tag_call_file, (N_API, N_TAG))

        hyperedge_rows = []
        hyperedge_cols = []
        hyperedge_values = []

        hyperedge_idx = 0

        # ---- 对每个 mashup 构造超边 ----
        for mashup in range(N_MASHUP):
            apis = np.where(mashup_api_matrix[mashup] > 0)[0]
            tags = np.where(mashup_tag_matrix[mashup] > 0)[0]

            hyperedge_rows.append(mashup)
            hyperedge_cols.append(hyperedge_idx)
            hyperedge_values.append(1.0)

            for api in apis:
                global_api_id = N_MASHUP + api
                hyperedge_rows.append(global_api_id)
                hyperedge_cols.append(hyperedge_idx)
                hyperedge_values.append(1)

            for tag in tags:
                global_tag_id = N_MASHUP + N_API + tag
                hyperedge_rows.append(global_tag_id)
                hyperedge_cols.append(hyperedge_idx)
                hyperedge_values.append(1)

            hyperedge_idx += 1

        # ---- 对每个 API 构造超边 ----
        for api in range(N_API):
            tags = np.where(api_tag_matrix[api] > 0)[0]

            global_api_id = N_MASHUP + api
            hyperedge_rows.append(global_api_id)
            hyperedge_cols.append(hyperedge_idx)
            hyperedge_values.append(1.0)

            for tag in tags:
                global_tag_id = N_MASHUP + N_API + tag
                hyperedge_rows.append(global_tag_id)
                hyperedge_cols.append(hyperedge_idx)
                hyperedge_values.append(1.0)

            hyperedge_idx += 1

        N_total = N_MASHUP + N_API + N_TAG
        N_edges = hyperedge_idx

        hyperedge_matrix = sp.coo_matrix(
            (hyperedge_values, (hyperedge_rows, hyperedge_cols)),
            shape=(N_total, N_edges)
        )

        hyperedge_T = hyperedge_matrix.T
        Dv, De = self.get_D_inv(hyperedge_T)

        Dv_torch = self.sparse_mx_to_torch_sparse_tensor(Dv)
        De_torch = self.sparse_mx_to_torch_sparse_tensor(De)
        hyperedge_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge_T)
        hyperedge_T_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge_matrix)

        temp = sparse.mm(Dv_torch, hyperedge_torch)
        normalized_1 = sparse.mm(temp, De_torch)
        normalized_2 = hyperedge_T_torch

        setattr(self, 'HGlobalEdge', hyperedge_torch)
        setattr(self, 'HGlobalEdge_T', hyperedge_T_torch)
        setattr(self, 'norm_global1', normalized_1)
        setattr(self, 'norm_global2', normalized_2)

        # print(f"Number of global hyperedges: {N_edges}")

        return normalized_1, normalized_2

    def load_tag_labels(self, file_path, num_entity, num_tags):
        """
        Args:
            file_path: mashup_tags.csv 或 api_tags.csv
            num_entity: N_MASHUP 或 N_API
            num_tags: N_TAG
        Returns:
            labels: FloatTensor shape [num_entity, num_tags], 0/1表示该entity是否包含该tag
        """
        labels = np.zeros((num_entity, num_tags), dtype=np.float32)
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [item for item in line.split(" ") if item != '']
                entity_id = int(parts[0])
                tag_ids = [int(t) for t in parts[1:]]
                for tag_id in tag_ids:
                    if 0 <= tag_id < num_tags:
                        labels[entity_id, tag_id] = 1.0
        return torch.FloatTensor(labels)

    def set_lr_scheduler(self):  # lr_scheduler：学习率调度器
        fac = lamb
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        # 每次的lr值：来自优化器的初始lr乘上一个lambda
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        payload = {
            'model': self.model.state_dict(),
        }
        if self.temporal_model is not None:
            payload['temporal_hgnn'] = self.temporal_model.state_dict()
        if self.edge_diffusion is not None:
            payload['edge_diffusion'] = self.edge_diffusion.state_dict()
            payload['edge_generation_threshold'] = float(self.edge_generation_threshold)
        torch.save(payload, self.weights_save_path)

    def load_model(self):
        state = torch.load(self.weights_save_path, map_location=self.device)

        if isinstance(state, dict) and 'model' in state:
            self.model.load_state_dict(state['model'])

            if self.temporal_model is not None and 'temporal_hgnn' in state:
                self.temporal_model.load_state_dict(state['temporal_hgnn'])

            if self.edge_diffusion is not None and 'edge_diffusion' in state:
                self.edge_diffusion.load_state_dict(state['edge_diffusion'])

            if 'edge_generation_threshold' in state:
                self.edge_generation_threshold = float(state['edge_generation_threshold'])
            return

        # Backward compatibility with old checkpoints that only saved model.state_dict().
        self.model.load_state_dict(state)

    def train(self):
        patience = args.patience  # 容忍多少个 epoch 没提升
        no_improve_count = 0
        best_loss = float('inf')  # 初始化最优 loss
        best_epoch = -1  # 记录最佳 epoch
        prev_epoch_avg_loss = None
        loss_plateau_count = 0

        for epoch in range(args.epoch):
            t1 = time()

            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            mashup_cl_loss, api_cl_loss = 0.0, 0.0
            mashup_global_cl_loss, api_global_cl_loss = 0.0, 0.0
            tag_mashup_loss, tag_api_loss = 0.0, 0.0
            edge_diff_loss = 0.0
            edge_pairs_seen = 0
            n_batch = data_generator.n_train // args.batch_size + 1
            valid_batches = 0
            sample_time = 0.
            forward_time = 0.
            backward_time = 0.
            bad_batch_count = 0
            mashup_emb_norm_sum = 0.0
            api_emb_norm_sum = 0.0
            temporal_m_norm_last = 0.0
            temporal_a_norm_last = 0.0

            for idx in range(n_batch):
                self.model.train()  # 模型为训练模式，像Dropout，Normalize这些层就会起作用，测试模式不会
                if self.temporal_model is not None:
                    self.temporal_model.train()
                if self.edge_diffusion is not None:
                    self.edge_diffusion.train()

                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()  # 采样正相关与负相关的物品id
                sample_time += time() - sample_t1

                if not (len(users) == len(pos_items) == len(neg_items)) or len(users) == 0:
                    bad_batch_count += 1
                    print(
                        f"[BatchCheck][WARN] Skip invalid batch at epoch={epoch}, batch={idx}: "
                        f"users={len(users)}, pos={len(pos_items)}, neg={len(neg_items)}"
                    )
                    continue

                if (
                    min(users) < 0 or max(users) >= self.n_mashup
                    or min(pos_items) < 0 or max(pos_items) >= self.n_api
                    or min(neg_items) < 0 or max(neg_items) >= self.n_api
                ):
                    bad_batch_count += 1
                    print(
                        f"[BatchCheck][WARN] Skip out-of-range ids at epoch={epoch}, batch={idx}."
                    )
                    continue

                forward_t1 = time()
                try:
                    outputs = self._forward_hgnn()
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    if not self._is_cuda_oom_like_error(e):
                        raise

                    bad_batch_count += 1
                    stage = f"epoch={epoch}, batch={idx}, phase=forward"
                    print(f"[OOM][WARN] {stage}: {e}")
                    self.optimizer.zero_grad(set_to_none=True)

                    recovered = self._reduce_temporal_memory_after_oom(stage)
                    if not recovered:
                        print(f"[OOM][WARN] {stage}: no additional temporal-memory fallback available; skipping batch.")
                        if self.device.type == 'cuda' and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    try:
                        outputs = self._forward_hgnn()
                    except (torch.OutOfMemoryError, RuntimeError) as e_retry:
                        if not self._is_cuda_oom_like_error(e_retry):
                            raise

                        print(f"[OOM][WARN] {stage}: retry after temporal backoff still failed; skipping batch. {e_retry}")
                        if self.device.type == 'cuda' and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                forward_time += time() - forward_t1

                if len(outputs) < 12:
                    raise RuntimeError(
                        f"Unexpected HGNN output length={len(outputs)}. "
                        "Temporal + contrastive + multitask training expects at least 12 values."
                    )

                (m_embeddings, a_embeddings,
                 m_c_proj, m_t_proj, a_c_proj, a_t_proj,
                 m_local_proj, a_local_proj, m_global_proj, a_global_proj,
                 mashup_tag_logits, api_tag_logits) = outputs[:12]

                mashup_emb_norm_sum += float(torch.norm(m_embeddings, dim=1).mean().detach().cpu())
                api_emb_norm_sum += float(torch.norm(a_embeddings, dim=1).mean().detach().cpu())
                if self.temporal_model is not None and self.temporal_model.latest_norm_stats:
                    temporal_stat = self.temporal_model.latest_norm_stats[-1]
                    temporal_m_norm_last = float(temporal_stat.get('mashup_norm_mean', 0.0))
                    temporal_a_norm_last = float(temporal_stat.get('api_norm_mean', 0.0))

                batch_mashup_tag_loss = self.tag_loss_fn(mashup_tag_logits, self.mashup_tag_labels)
                batch_api_tag_loss = self.tag_loss_fn(api_tag_logits, self.api_tag_labels)

                batch_edge_diff_loss = torch.tensor(0.0, device=self.device)
                edge_batch_stats = {'num_pairs': 0}
                if self.edge_diffusion is not None and self.edge_diffusion_weight > 0.0:
                    edge_m_embeddings = m_embeddings.detach() if args.edge_diffusion_detach_embeddings == 1 else m_embeddings
                    edge_a_embeddings = a_embeddings.detach() if args.edge_diffusion_detach_embeddings == 1 else a_embeddings
                    batch_edge_diff_loss, edge_batch_stats = self.edge_diffusion.edge_loss(
                        mashup_embeddings=edge_m_embeddings,
                        api_embeddings=edge_a_embeddings,
                        real_edge_index=self.real_edge_index,
                        negative_ratio=args.edge_diffusion_negative_ratio,
                        max_pos_edges=args.edge_diffusion_max_pos_per_batch,
                    )

                u_g_embeddings = m_embeddings[users]
                pos_i_g_embeddings = a_embeddings[pos_items]
                neg_i_g_embeddings = a_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                # contrastive loss for mashup: call vs tag
                batch_mashup_cl_loss = self.contrastive_loss(m_c_proj, m_t_proj, temperature=0.5)

                # contrastive loss for api: call vs tag
                batch_api_cl_loss = self.contrastive_loss(a_c_proj, a_t_proj, temperature=0.5)

                # contrastive loss for mashup: local vs global
                batch_mashup_global_cl_loss = self.contrastive_loss(m_local_proj, m_global_proj, temperature=0.5)

                # contrastive loss for api: local vs global
                batch_api_global_cl_loss = self.contrastive_loss(a_local_proj, a_global_proj, temperature=0.5)

                # 可选加权
                cl_loss_weight = args.beta  # 可以通过 args 传入更灵活

                batch_loss = (batch_mf_loss + batch_emb_loss + batch_reg_loss)
                batch_loss += cl_loss_weight * batch_mashup_cl_loss + cl_loss_weight * batch_api_cl_loss
                batch_loss += cl_loss_weight * batch_mashup_global_cl_loss + cl_loss_weight * batch_api_global_cl_loss
                batch_loss += batch_mashup_tag_loss + batch_api_tag_loss
                batch_loss += self.edge_diffusion_weight * batch_edge_diff_loss

                if not torch.isfinite(batch_loss):
                    print(
                        f"WARNING: non-finite batch loss at epoch={epoch}, batch={idx}; skipping batch. "
                        f"mf={float(batch_mf_loss):.6f}, emb={float(batch_emb_loss):.6f}, reg={float(batch_reg_loss):.6f}, "
                        f"m_cl={float(batch_mashup_cl_loss):.6f}, a_cl={float(batch_api_cl_loss):.6f}, "
                        f"m_g_cl={float(batch_mashup_global_cl_loss):.6f}, a_g_cl={float(batch_api_global_cl_loss):.6f}, "
                        f"m_tag={float(batch_mashup_tag_loss):.6f}, a_tag={float(batch_api_tag_loss):.6f}, "
                        f"edge_diff={float(batch_edge_diff_loss):.6f}"
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                backward_t1 = time()
                try:
                    batch_loss.backward()
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    if not self._is_cuda_oom_like_error(e):
                        raise

                    bad_batch_count += 1
                    stage = f"epoch={epoch}, batch={idx}, phase=backward"
                    print(f"[OOM][WARN] {stage}: {e}")
                    self.optimizer.zero_grad(set_to_none=True)
                    recovered = self._reduce_temporal_memory_after_oom(stage)
                    if recovered:
                        print(f"[OOM][WARN] {stage}: temporal-memory backoff applied; continuing with next batch.")
                    else:
                        print(f"[OOM][WARN] {stage}: no additional temporal-memory fallback available; continuing with next batch.")
                    if self.device.type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                if self.temporal_model is not None:
                    grad_is_healthy, bad_name = self.temporal_model.gradient_health_check()
                    if not grad_is_healthy:
                        bad_batch_count += 1
                        print(
                            f"[GradCheck][WARN] Non-finite temporal/model gradient at epoch={epoch}, batch={idx}, "
                            f"param={bad_name}; skipping optimizer step."
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    self.temporal_model.clip_gradients(max_norm=5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                if self.edge_diffusion is not None:
                    torch.nn.utils.clip_grad_norm_(self.edge_diffusion.parameters(), max_norm=5.0)

                    edge_grad_bad = False
                    for p in self.edge_diffusion.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            edge_grad_bad = True
                            break
                    if edge_grad_bad:
                        bad_batch_count += 1
                        print(
                            f"[GradCheck][WARN] Non-finite diffusion gradients at epoch={epoch}, batch={idx}; "
                            "skipping optimizer step."
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                self.optimizer.step()
                backward_time += time() - backward_t1
                valid_batches += 1

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

                mashup_cl_loss += float(batch_mashup_cl_loss)
                api_cl_loss += float(batch_api_cl_loss)

                mashup_global_cl_loss += float(batch_mashup_global_cl_loss)
                api_global_cl_loss += float(batch_api_global_cl_loss)

                tag_mashup_loss += float(batch_mashup_tag_loss)
                tag_api_loss += float(batch_api_tag_loss)
                edge_diff_loss += float(batch_edge_diff_loss)
                edge_pairs_seen += int(edge_batch_stats.get('num_pairs', 0))

            avg_epoch_loss = loss / max(1, valid_batches)
            if prev_epoch_avg_loss is not None:
                if avg_epoch_loss >= prev_epoch_avg_loss - 1e-8:
                    loss_plateau_count += 1
                else:
                    loss_plateau_count = 0

                if loss_plateau_count >= max(1, int(args.loss_plateau_patience)):
                    print(
                        f"[LossTrend][WARN] Epoch-average loss has not improved for "
                        f"{loss_plateau_count} epochs. current={avg_epoch_loss:.6f}, prev={prev_epoch_avg_loss:.6f}"
                    )
                    loss_plateau_count = 0
            prev_epoch_avg_loss = avg_epoch_loss

            if epoch > 150:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    no_improve_count = 0
                    best_epoch = epoch
                    if args.save_flag == 1:
                        self.save_model()
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(
                        f"⏹️ Early stopping triggered at epoch {epoch}. Best loss = {best_loss:.5f} (epoch {best_epoch})")
                    break

            self.lr_scheduler.step()  # 学习率更新

            if (
                self.edge_diffusion is not None
                and self.edge_refresh_interval > 0
                and (epoch + 1) >= max(0, int(args.edge_gen_warmup_epochs))
                and (epoch + 1) % self.edge_refresh_interval == 0
            ):
                try:
                    self._refresh_augmented_hypergraph(epoch + 1)
                except Exception as e:
                    print(f"[EdgeDiffusion][WARN] Graph augmentation failed at epoch {epoch + 1}: {e}")

            avg_forward_time = forward_time / max(1, valid_batches)
            if self.use_temporal and avg_forward_time > float(args.temporal_slowdown_warn_sec):
                print(
                    f"[Perf][WARN] Slow temporal forward at epoch={epoch}: avg_forward={avg_forward_time:.3f}s "
                    f"(steps={self.last_temporal_step_count}, stride={max(1, int(args.temporal_stride))})."
                )

            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    avg_m_emb_norm = mashup_emb_norm_sum / max(1, valid_batches)
                    avg_a_emb_norm = api_emb_norm_sum / max(1, valid_batches)
                    temporal_edges_str = self.temporal_call_edge_counts
                    epoch_time = max(1e-6, time() - t1)
                    batches_per_sec = valid_batches / epoch_time

                    perf_str = (
                        f"Epoch {epoch} [{epoch_time:.1f}s]: "
                        f"loss={avg_epoch_loss:.5f}, mf_loss={mf_loss:.5f}, "
                        f"mashup_cl_loss={mashup_cl_loss:.5f}, api_cl_loss={api_cl_loss:.5f}, "
                        f"mashup_global_cl_loss={mashup_global_cl_loss:.5f}, api_global_cl_loss={api_global_cl_loss:.5f}, "
                        f"tag_mashup_loss={tag_mashup_loss:.5f}, tag_api_loss={tag_api_loss:.5f}, "
                        f"edge_diff_loss={edge_diff_loss:.5f}, edge_pairs={edge_pairs_seen}, "
                        f"real_edges={int(self.real_edge_index.size(1))}, gen_edges={self.last_generated_edge_count}, "
                        f"aug_edges={self.last_augmented_edge_count}, edge_thr={self.edge_generation_threshold:.3f}, "
                        f"temporal_steps={self.last_temporal_step_count}, temporal_edge_counts={temporal_edges_str}, "
                        f"emb_norm_m={avg_m_emb_norm:.4f}, emb_norm_a={avg_a_emb_norm:.4f}, "
                        f"temp_norm_m={temporal_m_norm_last:.4f}, temp_norm_a={temporal_a_norm_last:.4f}, "
                        f"fwd_t={forward_time:.2f}s, bwd_t={backward_time:.2f}s, sample_t={sample_time:.2f}s, "
                        f"bad_batches={bad_batch_count}, batch_per_s={batches_per_sec:.2f}"
                    )
                    print(perf_str)

            if not math.isfinite(avg_epoch_loss):
                print('WARNING: epoch loss is non-finite (nan/inf). Continue to next epoch.')
                continue

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.save_flag == 1:
            self.save_model()
            print('save the weights in path: ', self.weights_save_path)

    def norm(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def cosine_similarity_scipy(self, vec1, vec2):
        """
        计算两个向量的余弦相似度

        参数:
        vec1 -- 第一个向量，NumPy 数组
        vec2 -- 第二个向量，NumPy 数组

        返回值:
        余弦相似度，浮点数
        """
        return 1 - cosine(vec1, vec2)

    def get_vector_by_id(self, id):
        vectors_file = resolve_embedding_file(args.embedding_base_dir, args.text_model, 'vectors.h5')
        with h5py.File(vectors_file, 'r') as f:
            if str(id) in f:
                return f[str(id)][:]
            else:
                return None

    def weighted_average(self, u_embeddings, similar):
        # 将 similar 转换为 PyTorch 张量
        similar = torch.tensor(similar, dtype=torch.float32, device=self.device)

        # 计算权重的总和
        total_weight = torch.sum(similar)

        # 将每个向量乘以其对应的权重
        weighted_embeddings = u_embeddings * similar.unsqueeze(1)

        # 对所有加权后的向量求和
        weighted_sum = torch.sum(weighted_embeddings, dim=0)

        # 将加权和除以权重的总和
        u_avg = weighted_sum / total_weight

        return u_avg


    # pos_items：正相关物品的id
    # neg_items：负相关物品的id
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # torch.mul():对应元素相乘
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)  # torch.mul():对应元素相乘

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def contrastive_loss(self, z1, z2, temperature=0.5):
        # 数值稳定版本：先清理非有限值，再用 logsumexp 避免 exp/log 下溢或上溢
        z1 = torch.nan_to_num(z1, nan=0.0, posinf=1e4, neginf=-1e4)
        z2 = torch.nan_to_num(z2, nan=0.0, posinf=1e4, neginf=-1e4)
        z1 = torch.clamp(z1, min=-1e4, max=1e4)
        z2 = torch.clamp(z2, min=-1e4, max=1e4)

        N = z1.size(0)
        if N == 0:
            return torch.tensor(0.0, device=z1.device)

        # 正样本距离
        pos_dist = torch.norm(z1 - z2, p=2, dim=1)  # [N]
        pos_dist = torch.clamp(pos_dist, min=0.0, max=1e6)
        sim_pos = -pos_dist / temperature

        # 计算 pairwise 距离矩阵
        z1_sq = (z1 ** 2).sum(dim=1).unsqueeze(1)  # [N, 1]
        z2_sq = (z2 ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
        dot_product = torch.matmul(z1, z2.T)  # [N, N]
        dist_squared = z1_sq + z2_sq - 2 * dot_product
        dist_squared = torch.clamp(dist_squared, min=1e-12, max=1e12)
        dist_matrix = torch.sqrt(dist_squared)  # [N, N]

        # 转为负相似度并屏蔽正样本
        logits_zw = -dist_matrix / temperature
        mask = torch.eye(N, dtype=torch.bool, device=z1.device)
        logits_zw = logits_zw.masked_fill(mask, float('-inf'))

        # InfoNCE: -sim_pos + log(exp(sim_pos) + sum exp(neg))，用 logsumexp 稳定计算
        all_logits = torch.cat([sim_pos.unsqueeze(1), logits_zw], dim=1)
        log_denom = torch.logsumexp(all_logits, dim=1)
        loss = log_denom - sim_pos
        return loss.mean()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape


def main():
    # 调用 parse_args 获取命令行参数
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_tags'] = data_generator.n_tags

    pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            Engine.train()
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
        Engine.train()


if __name__ == '__main__':
    main()