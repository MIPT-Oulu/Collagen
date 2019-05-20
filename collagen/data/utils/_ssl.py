from collagen.data import ItemLoader, AugmentedGroupSampler, DataProvider


def pimodel_data_provider(model, train_labeled_data, train_unlabeled_data, val_labeled_data, val_unlabeled_data,
                      transforms, parse_item, bs, num_threads, item_loaders=dict()):
    """
    Default setting of data provider for Pi-Model

    """
    item_loaders["labeled_train"] = ItemLoader(meta_data=train_labeled_data,
                                               transform=transforms[0],
                                               parse_item_cb=parse_item,
                                               batch_size=bs, num_workers=num_threads,
                                               shuffle=True)

    item_loaders["unlabeled_train"] = AugmentedGroupSampler(model=model,
                                                            meta_data=train_unlabeled_data,
                                                            n_augmentations=1,
                                                            augmentation=transforms[2],
                                                            transform=transforms[1],
                                                            parse_item_cb=parse_item,
                                                            batch_size=bs, num_workers=num_threads,
                                                            shuffle=True)

    item_loaders["labeled_eval"] = ItemLoader(meta_data=val_labeled_data,
                                             transform=transforms[1],
                                             parse_item_cb=parse_item,
                                             batch_size=bs, num_workers=num_threads,
                                             shuffle=False)

    item_loaders["unlabeled_eval"] = AugmentedGroupSampler(model=model,
                                                          meta_data=val_unlabeled_data,
                                                          n_augmentations=1,
                                                          augmentation=transforms[2],
                                                          transform=transforms[1],
                                                          parse_item_cb=parse_item,
                                                          batch_size=bs, num_workers=num_threads,
                                                          shuffle=False)

    return DataProvider(item_loaders)
