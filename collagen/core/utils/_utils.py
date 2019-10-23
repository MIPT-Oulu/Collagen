from typing import Tuple
import os
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP_APEX
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex ")

def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, required_grad=False, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.to('cpu').detach().numpy()
            elif required_grad:
                x_cpu = x.to('cpu')
            else:
                x_cpu = x.to('cpu').required_grad_(False)
        elif use_numpy:
            x_cpu = x.numpy()

    return x_cpu


def wrap_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    else:
        return x


def auto_detect_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_modules(modules: torch.nn.Module or Tuple[torch.nn.Module], invert=False):
    requires_grad = invert
    _modules = wrap_tuple(modules)
    for md in _modules:
        # md.train(requires_grad)
        for param in md.parameters():
            param.requires_grad = requires_grad


def init_dist_env():
    """Set variables for multiple processes to communicate between themselves"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '22222'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '0'


def convert_to_distributed(args, gpu, ngpus, d_network, g_network, d_optim, g_optim):
    # rank will be necessary in future for cluster computing, for now we will settle for gpu
    args.rank = int(os.environ['RANK']) * ngpus + gpu
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank,
                            init_method='env://')
    print('Distributed Init done on GPU:', args.gpu)
    # we will set the benchmark to true so that the pytorch's build-in auto tuner will find the best algorithm
    # depending on the hardware under the OS
    cudnn.benchmark = True
    # set the current device, remember for different spawned process, args.gpu would be different
    torch.cuda.set_device(args.gpu)
    # split the batch size along gpu(s), it is a good practise to use a batch size multiple of the number of gpu(s)
    args.batch_size = int(args.batch_size / ngpus)

    if args.use_apex:
        # if your network does not have any batchnorm layer, don't use syncnb
        d_network = apex.parallel.convert_syncbn_model(d_network)
        g_network = apex.parallel.convert_syncbn_model(g_network)
    # d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
    # g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
    if args.use_apex:
        [d_network, g_network], [d_optim, g_optim] = amp.initialize([d_network, g_network], [d_optim, g_optim],
                                                                    opt_level='O1', loss_scale=args.loss_scale)
        d_network = DDP_APEX(d_network, delay_allreduce=True)
        g_network = DDP_APEX(g_network, delay_allreduce=True)
    else:
        d_network = DDP(d_network, device_ids=[gpu])
        g_network = DDP(g_network, device_ids=[gpu])
    return args, d_network, g_network, d_optim, g_optim