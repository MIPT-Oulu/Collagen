import os
import warnings

import torch
import yaml
from torch import distributed as dist, multiprocessing as mp
from torch.backends import cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from collagen import Module

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP_APEX
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex ")

__all__ = ["init_dist_env", "convert_according_to_args", "convert_to_distributed", "first_gpu_or_cpu_in_use",
           "kick_off_launcher"]


def init_dist_env():
    """Set variables for multiple processes to communicate between themselves"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '22222'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '0'


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
    rank = int(os.environ['RANK']) * ngpus + gpu
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=rank,
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
        init_dist_env()
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
