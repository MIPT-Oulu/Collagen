import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation Batch Size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--bw', type=int, default=64, help='Bandwidth of model')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--snapshots', default='snapshots', help='Where to save the snapshots')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--mms', type=bool, default=False, help='Whether to use MultiModelStrategy')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="cnn", help='Comment of log')
    parser.add_argument('--n_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--distributed', type=bool, default=True, help='whether to use DDP')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--use_apex', default=True, type=bool, help='Whether to use apex library')
    parser.add_argument('--loss_scale', default=None, type=int, help='loss scale for apex amp')
    parser.add_argument('--opt_level', default='O1', type=str, help='Optimization level for amp')
    parser.add_argument('--shell_launch', default=True, type=bool, help='Shell launched program is not debuggable in'
                                                                        'PyCharm, set it false to use '
                                                                        'pytorch.multiprocessing which is debuggable')
    parser.add_argument('--local_rank', default=0, type=int, help='rank of the gpu within the node')
    parser.add_argument('--world_size', default=2, type=int, help='world_size = number_of_nodes*gpus_per_node')
    parser.add_argument('--ngpus_per_node', default=2, type=int, help='Number of gpus you want to use')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str, help='Master Address for backend')
    parser.add_argument('--master_port', default=None, type=str, help='Master port for backend')
    parser.add_argument('--suppress_warning', default=True, type=bool, help='whether to print warning messages')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers, too many cooks spoil the broth')

    args = parser.parse_args()

    return args
