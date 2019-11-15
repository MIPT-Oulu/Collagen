from typing import Tuple
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from collagen.core import Module
import socket
from contextlib import closing


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


def init_dist_env(world_size, master_addr='127.0.0.1', master_port=None ):
    """Set variables for multiple processes to communicate between themselves"""
    os.environ['MASTER_ADDR'] = str(master_addr)
    if master_port is None:
        os.environ['MASTER_PORT'] = str(find_free_host_port(host=master_addr))
    else:
        os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = '0'
    os.environ['OMP_NUM_THREADS'] = '2'


def convert_according_to_args(args, gpu, ngpus, network, optim):
    from apex import amp
    # os.environ['RANK'] = str(gpu)
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


def convert_to_apex_ddp(args, gpu, ngpus, network, optim):
    import apex
    from apex.parallel import DistributedDataParallel as DDP_APEX
    from apex import amp
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


def convert_to_distributed(args, gpu, ngpus, network, optim):
    import apex
    from apex.parallel import DistributedDataParallel as DDP_APEX
    from apex import amp
    # rank will be necessary in future for cluster computing, for now we will settle for gpu
    args.local_rank = gpu
    dist.init_process_group(backend=args.dist_backend, rank=args.local_rank, world_size=args.world_size, init_method='env://')
    print('Distributed Init done on GPU:', args.local_rank)
    # we will set the benchmark to true so that the pytorch's build-in auto tuner will find the best algorithm
    # depending on the hardware under the OS
    cudnn.benchmark = True
    # set the current device, remember for different spawned process, args.gpu would be different
    torch.cuda.set_device(args.local_rank)
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


def kick_off_launcher(args, sampling_config: dict, strategy_config: dict, worker_process):
    """The main function to create process(es) according to necessity and passed arguments"""
    assert hasattr(args, 'distributed') == True, 'The argument `distributed` is missing'
    if args.distributed:
        assert hasattr(args, 'world_size') == True, 'The argument `world_size` is missing'
        init_dist_env(args.world_size)
    # get the number of gpus
    assert hasattr(args, 'ngpus_per_node') == True, 'The argument `ngpus_per_node` is missing'
    if args.ngpus_per_node is not None:
        ngpus = args.ngpus_per_node
    else:
        if torch.cuda.is_available():
            ngpus = torch.cuda.device_count()
        else:
            ngpus = 0
    assert ngpus > 1 and args.distributed, 'You need multiple gpus to do distributed computing, ' \
                                             'set distributed to false'
    if args.distributed:
        # we one process per gpu for distributed setting
        mp.spawn(worker_process, nprocs=ngpus, args=(ngpus, sampling_config, strategy_config, args))
    else:
        worker_process(args.local_rank, ngpus, sampling_config, strategy_config, args)


def first_gpu_or_cpu_in_use(device):
    # device is gpu ordinal, and is the first gpu
    ordinal_first = isinstance(device, int) and device == 0
    string_first = isinstance(device, str) and (device == 'cuda:0' or device == 'cpu')
    device_first = isinstance(device, torch.device) and (device == torch.device('cuda:0') or
                                                         device == torch.device('cpu'))
    return ordinal_first or string_first or device_first


def find_free_host_port(host):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))  # bind to host
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # socket level access ensured by SOL_SOCKET
        return s.getsockname()[1]
