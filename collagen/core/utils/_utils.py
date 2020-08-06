from typing import Tuple
from collections import OrderedDict
import torch
import yaml
import os

__all__ = ["to_cpu", "wrap_tuple", "auto_detect_device", "freeze_modules", "create_default_config"]


def create_default_config(root, model_name="mymodel"):
    # Create config.yaml
    config_dict = dict()
    config_dict['defaults'] = [{'arg': 'args1'}, {'sampling': 'sampling1'}]
    config_dict['strategy'] = {'stage_names': ['train', 'eval'], 'model_names': [f"{model_name}"],
                               'train_model_names': [f"{model_name}"],
                               'eval_model_names': [f"{model_name}"],
                               'accumulate_grad': {f"{model_name}": False},
                               'train_starts_at_epoch': {f"{model_name}": 0}}

    # Create args1.yaml
    args_dict = dict()
    args_dict['n_epochs'] = 100
    args_dict['bs'] = 128
    args_dict['dropout'] = 0.5
    args_dict['bw'] = 64
    args_dict['wd'] = 1e-4
    args_dict['lr'] = 1e-4
    args_dict['num_workers'] = 4
    args_dict['snapshots'] = 'snapshots'
    args_dict['seed'] = 12345
    args_dict['dataset'] = 'mnist'
    args_dict['data_dir'] = './data'
    args_dict['log_dir'] = ''
    args_dict['comment'] = 'mycomment'

    # Create
    sampling_dict = dict()
    loader_dict = {'batches_per_iter': 1, 'data_key': 'data', 'target_key': 'target'}
    train_data_provider_dict = {'data_provider': {f'{model_name}': {'loader_train': loader_dict.copy()}}}
    eval_data_provider_dict = {'data_provider': {f'{model_name}': {'loader_eval': loader_dict.copy()}}}
    sampling_dict['sampling'] = {'train': train_data_provider_dict, 'eval': eval_data_provider_dict}

    os.makedirs(os.path.join(root, 'args'))
    os.makedirs(os.path.join(root, 'sampling'))

    config_fullname = os.path.join(root, 'config.yaml')
    args_fullname = os.path.join(root, 'args', 'args1.yaml')
    sampling_fullname = os.path.join(root, 'sampling', 'sampling1.yaml')

    print(f'Creating file {config_fullname}')
    with open(config_fullname, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f'Creating file {args_fullname}')
    with open(args_fullname, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False )

    print(f'Creating file {sampling_fullname}')
    with open(sampling_fullname, 'w') as f:
        yaml.dump(sampling_dict, f, default_flow_style=False, sort_keys=False)


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
            if x.requires_grad:
                x_cpu = x.detach().numpy()
            else:
                x_cpu = x.numpy()

    return x_cpu


def wrap_tuple(x):
    if hasattr(x, '__len__') and not isinstance(x, str):
        return x
    else:
        return (x,)


def auto_detect_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_modules(modules: torch.nn.Module or Tuple[torch.nn.Module], invert=False):
    requires_grad = invert
    _modules = wrap_tuple(modules)
    for md in _modules:
        # md.train(requires_grad)
        for param in md.parameters():
            param.requires_grad = requires_grad
