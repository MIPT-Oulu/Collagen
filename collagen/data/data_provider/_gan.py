from collagen.data.samplers import ItemLoader, GANFakeSampler, GaussianNoiseSampler, DistributedGANFakeSampler
from collagen.data import DataProvider, DistributedItemLoader
import torch

__all__ = ["gan_data_provider", "distributed_gan_data_provider"]


def gan_data_provider(g_network, item_loaders, train_ds, classes,
                      latent_size, transforms, parse_item, bs, num_threads, device):
    """
    Default setting of data provider for GAN

    """
    item_loaders['real'] = ItemLoader(meta_data=train_ds,
                                      transform=transforms,
                                      parse_item_cb=parse_item,
                                      batch_size=bs, num_workers=num_threads,
                                      shuffle=True)

    item_loaders['fake'] = GANFakeSampler(g_network=g_network,
                                          batch_size=bs,
                                          latent_size=latent_size)

    item_loaders['noise'] = GaussianNoiseSampler(batch_size=bs,
                                                 latent_size=latent_size,
                                                 device=device, n_classes=len(classes))

    return DataProvider(item_loaders)


def distributed_gan_data_provider(g_network, item_loaders, train_ds, classes,
                      latent_size, transforms, parse_item, args):
    """
    Default setting of data provider for GAN

    """
    bs = args.batch_size
    num_threads = args.workers
    gpu = args.gpu
    distributed = args.distributed
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    if distributed:
        item_loaders['real'] = DistributedItemLoader(meta_data=train_ds,
                                                     transform=transforms,
                                                     parse_item_cb=parse_item,
                                                     args=args)
        item_loaders['fake'] = DistributedGANFakeSampler(g_network=g_network, batch_size=bs,
                                                         latent_size=latent_size, gpu=gpu)

        item_loaders['noise'] = GaussianNoiseSampler(batch_size=bs,
                                                     latent_size=latent_size,
                                                     device=gpu, n_classes=len(classes))

    else:
        item_loaders['real'] = ItemLoader(meta_data=train_ds,
                                          transform=transforms,
                                          parse_item_cb=parse_item,
                                          batch_size=bs, num_workers=num_threads,
                                          shuffle=True)

        item_loaders['fake'] = GANFakeSampler(g_network=g_network,
                                              batch_size=bs,
                                              latent_size=latent_size)

        item_loaders['noise'] = GaussianNoiseSampler(batch_size=bs,
                                                     latent_size=latent_size,
                                                     device=device, n_classes=len(classes))

    return DataProvider(item_loaders)
