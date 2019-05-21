from collagen.data import ItemLoader, GANFakeSampler, GaussianNoiseSampler, DataProvider


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
