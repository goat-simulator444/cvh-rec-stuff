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
    return parser.parse_args()
