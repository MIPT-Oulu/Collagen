import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="dcgan", help='Comment of log')
    parser.add_argument('--grid_shape', type=int, default=8, help='Shape of grid of generated images')
    parser.add_argument('--mms', type=bool, default=True, help='Shape of grid of generated images')

    parser.add_argument('--distributed', type=bool, default=True, help='whether to use DDP')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--use_apex', default=True, type=bool, help='Whether to use apex library')
    parser.add_argument('--loss_scale', default=None, type=int, help='loss scale for apex amp')
    parser.add_argument('--n_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--opt_level', default='O1', type=str, help='number of input channels')
    parser.add_argument('--local_rank', default=0, type=int, help='rank of the gpu within the node')
    parser.add_argument('--world_size', default=2, type=int, help='world_size = number_of_nodes*gpus_per_node')
    parser.add_argument('--ngpus_per_node', default=2, type=int, help='Number of gpus you want to use')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str, help='Master Address for backend')
    parser.add_argument('--master_port', default=None, type=str, help='Master port for backend')
    parser.add_argument('--suppress_warning', default=True, type=bool, help='whether to print warning messages')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers ')
    parser.add_argument('--shell_launch', default=True, type=bool, help='Shell launched program is not debuggable in'
                                                                        'PyCharm, set it false to use '
                                                                        'pytorch.multiprocessing which is debuggable')
    args = parser.parse_args()


    return args