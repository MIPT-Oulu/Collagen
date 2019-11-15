from collagen.core import Module
from collagen.data.samplers import ItemLoader, GANFakeSampler, GaussianNoiseSampler
from collagen.data import DataProvider
import torch
import pandas as pd


def gan_data_provider(g_network: Module,
                      item_loaders: dict,
                      train_ds: pd.DataFrame,
                      classes: list,
                      latent_size: int,
                      transform,
                      parse_item,
                      distributed=False,
                      shuffle=True,
                      pin_memory=False,
                      local_rank=0,
                      world_size=1,
                      batch_size=32,
                      num_workers=1):
    """
    Default setting of data provider for GAN

    """


    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(local_rank))
    else:
        device = torch.device('cpu')

    item_loaders['real'] = ItemLoader(meta_data=train_ds,
                                      transform=transform,
                                      parse_item_cb=parse_item,
                                      distributed=distributed,
                                      shuffle=(distributed and shuffle) or (not distributed),
                                      pin_memory=(not distributed and pin_memory) or distributed,
                                      local_rank=local_rank,
                                      world_size=world_size,
                                      batch_size=batch_size,
                                      num_workers=num_workers
                                      )

    item_loaders['fake'] = GANFakeSampler(g_network=g_network,
                                          batch_size=batch_size,
                                          latent_size=latent_size,
                                          distributed=distributed)

    item_loaders['noise'] = GaussianNoiseSampler(batch_size=batch_size,
                                                 latent_size=latent_size,
                                                 device=device, n_classes=len(classes),
                                                 distributed=distributed)

    return DataProvider(item_loaders)
