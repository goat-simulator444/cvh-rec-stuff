import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run HCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--alpha1', type=float, default=0.6,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--alpha2', type=float, default=0.6,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--alpha3', type=float, default=0.6,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--alpha4', type=float, default=0.6,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='The weighting of contrastive learning')
    parser.add_argument('--similar_threthold', type=float, default=5,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--data_path', nargs='?', default='dataset/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default=r'',
                        help='Project path.')
    parser.add_argument('--save_recom', type=int, default=1,
                        help='Whether save the recommendation results.')
    parser.add_argument('--dataset', nargs='?', default='fold1',
                        help='Choose a dataset from given folder')
    parser.add_argument('--method', nargs='?', default='CVH-REC')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--seed', type=int, default=85)
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of output loss value.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=768,
                        help='Embedding size.')
    parser.add_argument('--text_model', nargs='?', default='mpnet',
                        help='Text embedding model alias or HF name (e.g., mpnet, bge-base, bge-large, e5-base, e5-large, gte-base, gte-large).')
    parser.add_argument('--use_bge', type=int, default=0,
                        help='Ablation switch for BGE embeddings: 1 forces text_model to bge-large, 0 keeps text_model as provided.')
    parser.add_argument('--embedding_base_dir', nargs='?', default='data',
                        help='Base directory for storing/loading text embedding h5 files.')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate.')
    parser.add_argument('--model_type', nargs='?', default='hcf',
                        help='Specify the name of model (hcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm_adj',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='hcf')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. ')
    #probability of an element to be zeroed.
    parser.add_argument('--drop_edge',type=float,default=0.95,
                        help="perserve the percent of edges")
    parser.add_argument('--Ks', nargs='?', default='[5, 20]',
                        help='Output sizes of every layer')
    parser.add_argument('--output', nargs='?', default='../output',
                        help='Output sizes of every layer')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--temporal_hypergraph', type=int, default=1,
                        help='1: Use TE_temporal.csv when available and build per-time-step sub-hypergraphs, 0: static hypergraph only.')
    parser.add_argument('--temporal_window_size', type=float, default=64.0,
                        help='Temporal window size used to split interactions into time slices when temporal hypergraph is enabled.')
    parser.add_argument('--temporal_interaction_file', nargs='?', default='auto',
                        help='Temporal interaction file name under dataset fold (e.g., TE_temporal.csv). Use auto to detect TE_temporal.csv then fallback to TE.csv.')
    parser.add_argument('--temporal_timestamp_fallback', nargs='?', default='line_order',
                        choices=['line_order', 'zero'],
                        help='Timestamp fallback method when temporal file is missing timestamps.')
    parser.add_argument('--use_temporal', type=int, default=1,
                        help='1: use TemporalHGNN sequence modeling, 0: use static HGNN forward only.')
    parser.add_argument('--temporal_max_steps', type=int, default=0,
                        help='Maximum temporal steps kept in sequence (0 means all available steps).')
    parser.add_argument('--temporal_stride', type=int, default=1,
                        help='Use every n-th temporal step to reduce slowdown (1 means no downsampling).')
    parser.add_argument('--temporal_debug_logs', type=int, default=0,
                        help='1: log per-step temporal embedding norms from TemporalHGNN.')
    parser.add_argument('--temporal_debug_log_every', type=int, default=1,
                        help='Temporal norm logging interval when temporal_debug_logs=1.')
    parser.add_argument('--temporal_slowdown_warn_sec', type=float, default=1.2,
                        help='Warn when average forward pass time per batch exceeds this value (seconds).')
    parser.add_argument('--loss_plateau_patience', type=int, default=8,
                        help='Warn when epoch-average loss does not improve for N consecutive epochs.')

    parser.add_argument('--use_diffusion', type=int, default=1,
                        help='1: enable diffusion edge augmentation in training loop, 0: disable.')
    parser.add_argument('--edge_diffusion', type=int, default=1,
                        help='1: enable diffusion-based edge generator, 0: disable.')
    parser.add_argument('--edge_diffusion_weight', type=float, default=0.05,
                        help='Loss weight for diffusion edge prediction objective.')
    parser.add_argument('--edge_diffusion_negative_ratio', type=float, default=1.0,
                        help='Negative sampling ratio for edge diffusion training.')
    parser.add_argument('--edge_diffusion_hidden_dim', type=int, default=256,
                        help='Hidden size of the diffusion edge generator.')
    parser.add_argument('--edge_diffusion_steps', type=int, default=8,
                        help='Number of diffusion steps used in edge generator.')
    parser.add_argument('--edge_diffusion_detach_embeddings', type=int, default=1,
                        help='1: detach recommender embeddings when training edge diffusion for stability.')
    parser.add_argument('--edge_diffusion_max_pos_per_batch', type=int, default=4096,
                        help='Max positive edges sampled per training batch for edge diffusion loss.')

    parser.add_argument('--edge_gen_threshold', type=float, default=0.80,
                        help='Base threshold for generated edge probabilities.')
    parser.add_argument('--edge_gen_threshold_step', type=float, default=0.02,
                        help='Threshold increment during automatic threshold tuning.')
    parser.add_argument('--edge_gen_threshold_tuning_steps', type=int, default=4,
                        help='Max threshold tuning steps to control over-generation.')
    parser.add_argument('--edge_gen_max_edges_per_node', type=int, default=16,
                        help='Hard cap on generated edges per mashup node.')
    parser.add_argument('--edge_gen_topk_per_node', type=int, default=256,
                        help='Top-k candidates per mashup evaluated before threshold filtering.')
    parser.add_argument('--edge_gen_max_total_edges', type=int, default=50000,
                        help='Hard cap on total generated edges per refresh.')
    parser.add_argument('--edge_gen_refresh_interval', type=int, default=10,
                        help='Refresh interval (epochs) for graph augmentation using generated edges.')
    parser.add_argument('--edge_gen_warmup_epochs', type=int, default=10,
                        help='Warmup epochs before first graph augmentation refresh.')
    parser.add_argument('--edge_gen_chunk_size', type=int, default=4096,
                        help='Chunk size when scoring all APIs per mashup during edge generation.')
    args = parser.parse_args()

    # Normalize ablation flags to {0, 1} to keep behavior consistent across scripts.
    args.use_bge = 1 if int(args.use_bge) == 1 else 0
    args.use_temporal = 1 if int(args.use_temporal) == 1 else 0
    args.use_diffusion = 1 if int(args.use_diffusion) == 1 else 0

    # Make ablation switch authoritative for BGE experiments.
    if args.use_bge == 1:
        args.text_model = 'bge-large'

    return args
