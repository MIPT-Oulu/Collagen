from typing import Tuple
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import warnings
from collagen.core import Module
import socket
from contextlib import closing

import apex
from apex.parallel import DistributedDataParallel as DDP_APEX
from apex import amp



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


def init_dist_env(world_size):
    """Set variables for multiple processes to communicate between themselves"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(find_free_localhost_port())
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = '0'
    os.environ['OMP_NUM_THREADS'] = '2'


def convert_according_to_args(args, gpu, ngpus, network, optim):
    if args.distributed:
        return convert_to_distributed(args, gpu, ngpus, network, optim)
    else:
        # for distributed setting, the convert_to_distributed function initialize the amp, but for single gpu
        # process user has to manually initialize the automated mixed precision
        if args.use_apex:
            network, optim = amp.initialize(network, optim,
                                            opt_level=args.opt_level,
                                            loss_scale=args.loss_scale,
                                            verbosity=not args.suppress_warning)
        return args, network, optim


def convert_to_distributed(args, gpu, ngpus, network, optim):
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
        if isinstance(network, list):
            for i in range(len(network)):
                network[i] = apex.parallel.convert_syncbn_model(network[i])
        elif isinstance(network, Module) or isinstance(network, torch.nn.Module):
            network = apex.parallel.convert_syncbn_model(network)

        network, optim = amp.initialize(network, optim,
                                        opt_level=args.opt_level, loss_scale=args.loss_scale,
                                        verbosity=not args.suppress_warning)

        if isinstance(network, list):
            for i in range(len(network)):
                network[i] = DDP_APEX(network[i], delay_allreduce=True)
        elif isinstance(network, Module) or isinstance(network, torch.nn.Module):
            network = DDP_APEX(network, delay_allreduce=True)

    else:
        if isinstance(network, list):
            for i in range(len(network)):
                network[i] = DDP(network[i], device_ids=[gpu])
        elif isinstance(network, Module) or isinstance(network, torch.nn.Module):
            network = DDP(network, device_ids=[gpu])
    return args, network, optim


def kick_off_launcher(args, worker_process):
    """The main function to create process(es) according to necessity and passed arguments"""
    if args.distributed:
        init_dist_env(args.world_size)
    if args.suppress_warning:
        warnings.filterwarnings("ignore")
    # set number of channels

    args.n_channels = 1
    if args.distributed:
        args.world_size = int(os.environ['WORLD_SIZE'])

    # load some yml files for sampling and strategy configuration
    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)
    if args.mms:
        with open("strategy.yml", "r") as f:
            strategy_config = yaml.load(f)
    else:
        strategy_config = None

    # get the number of gpus
    ngpus = torch.cuda.device_count()
    if args.distributed:
        # we one process per gpu for distributed setting
        mp.spawn(worker_process, nprocs=ngpus, args=(ngpus, sampling_config, strategy_config, args))
    else:
        worker_process(args.gpu, ngpus, sampling_config, strategy_config, args)


def first_gpu_or_cpu_in_use(device):
    # device is gpu ordinal, and is the first gpu
    ordinal_first = isinstance(device, int) and device == 0
    string_first = isinstance(device, str) and (device == 'cuda:0' or device == 'cpu')
    device_first = isinstance(device, torch.device) and (device == torch.device('cuda:0') or
                                                         device == torch.device('cpu'))
    return ordinal_first or string_first or device_first


def find_free_localhost_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('127.0.0.1', 0)) # bind to localhost
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # socket level access ensured by SOL_SOCKET
        return s.getsockname()[1]