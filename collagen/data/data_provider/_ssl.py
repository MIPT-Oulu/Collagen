from collagen.data import ItemLoader, AugmentedGroupSampler, AugmentedGroupStudentTeacherSampler, DataProvider


def pimodel_data_provider(model, train_labeled_data, train_unlabeled_data, val_labeled_data, val_unlabeled_data,
                          transforms, parse_item, bs, num_threads, item_loaders=dict(), root="", n_augmentations=1,
                          output_type='logits'):
    """
    Default setting of data provider for Pi-Model

    """
    # item_loaders["labeled_train"] = ItemLoader(root=root, meta_data=train_labeled_data, name='l',
    #                                            transform=transforms[0],
    #                                            parse_item_cb=parse_item,
    #                                            batch_size=bs, num_workers=num_threads,
    #                                            shuffle=True)

    item_loaders["labeled_train"] = AugmentedGroupSampler(root=root, model=model, name='l', output_type=output_type,
                                                          meta_data=train_labeled_data,
                                                          n_augmentations=n_augmentations,
                                                          augmentation=transforms[2],
                                                          transform=transforms[1],
                                                          parse_item_cb=parse_item,
                                                          batch_size=bs, num_workers=num_threads,
                                                          shuffle=True)

    item_loaders["unlabeled_train"] = AugmentedGroupSampler(root=root, model=model, name='u', output_type=output_type,
                                                            meta_data=train_unlabeled_data,
                                                            n_augmentations=n_augmentations,
                                                            augmentation=transforms[2],
                                                            transform=transforms[1],
                                                            parse_item_cb=parse_item,
                                                            batch_size=bs, num_workers=num_threads,
                                                            shuffle=True)

    # item_loaders["labeled_eval"] = ItemLoader(root=root, meta_data=val_labeled_data, name='l',
    #                                           transform=transforms[1],
    #                                           parse_item_cb=parse_item,
    #                                           batch_size=bs, num_workers=num_threads,
    #                                           shuffle=False)

    item_loaders["labeled_eval"] = AugmentedGroupSampler(root=root, model=model, name='l', output_type=output_type,
                                                         meta_data=val_labeled_data,
                                                         n_augmentations=n_augmentations,
                                                         augmentation=transforms[2],
                                                         transform=transforms[1],
                                                         parse_item_cb=parse_item,
                                                         batch_size=bs, num_workers=num_threads,
                                                         shuffle=False)

    item_loaders["unlabeled_eval"] = AugmentedGroupSampler(root=root, model=model, name='u', output_type=output_type,
                                                           meta_data=val_unlabeled_data,
                                                           n_augmentations=n_augmentations,
                                                           augmentation=transforms[2],
                                                           transform=transforms[1],
                                                           parse_item_cb=parse_item,
                                                           batch_size=bs, num_workers=num_threads,
                                                           shuffle=False)

    return DataProvider(item_loaders)


def mt_data_provider(st_model, te_model, train_labeled_data, train_unlabeled_data, val_labeled_data,
                     val_unlabeled_data,
                     transforms, parse_item, bs, num_threads, item_loaders=dict(), output_type='logits', root=""):
    """
    Default setting of data provider for Mean-Teacher

    """

    # Train
    item_loaders["labeled_train_st"] = AugmentedGroupSampler(root=root, name='l_st',
                                                             meta_data=train_labeled_data,
                                                             model=st_model,
                                                             n_augmentations=1,
                                                             augmentation=transforms[2],
                                                             transform=transforms[1],
                                                             parse_item_cb=parse_item,
                                                             batch_size=bs, num_workers=num_threads,
                                                             shuffle=True)

    item_loaders["unlabeled_train_st"] = AugmentedGroupSampler(root=root, name='u_st',
                                                               model=st_model,
                                                               meta_data=train_unlabeled_data,
                                                               n_augmentations=1,
                                                               augmentation=transforms[2],
                                                               transform=transforms[1],
                                                               parse_item_cb=parse_item,
                                                               batch_size=bs, num_workers=num_threads,
                                                               shuffle=True)

    item_loaders["labeled_train_te"] = AugmentedGroupSampler(root=root, name='l_te',
                                                             meta_data=train_labeled_data,
                                                             model=te_model,
                                                             n_augmentations=1,
                                                             augmentation=transforms[2],
                                                             transform=transforms[1],
                                                             parse_item_cb=parse_item,
                                                             batch_size=bs, num_workers=num_threads,
                                                             detach=True,
                                                             shuffle=True)

    item_loaders["unlabeled_train_te"] = AugmentedGroupSampler(root=root, name='u_te',
                                                               model=te_model,
                                                               meta_data=train_unlabeled_data,
                                                               n_augmentations=1,
                                                               augmentation=transforms[2],
                                                               transform=transforms[1],
                                                               parse_item_cb=parse_item,
                                                               batch_size=bs, num_workers=num_threads,
                                                               detach=True,
                                                               shuffle=True)

    # Eval

    item_loaders["labeled_eval_st"] = AugmentedGroupSampler(root=root, name='l_st',
                                                            meta_data=val_labeled_data,
                                                            model=st_model,
                                                            n_augmentations=1,
                                                            augmentation=transforms[2],
                                                            transform=transforms[1],
                                                            parse_item_cb=parse_item,
                                                            batch_size=bs, num_workers=num_threads,
                                                            shuffle=False)

    item_loaders["unlabeled_eval_st"] = AugmentedGroupSampler(root=root, name='u_st',
                                                              model=st_model,
                                                              meta_data=val_unlabeled_data,
                                                              n_augmentations=1,
                                                              augmentation=transforms[2],
                                                              transform=transforms[1],
                                                              parse_item_cb=parse_item,
                                                              batch_size=bs, num_workers=num_threads,
                                                              shuffle=False)

    item_loaders["labeled_eval_te"] = ItemLoader(root=root, meta_data=val_labeled_data, name='l_te_eval',
                                                 transform=transforms[1],
                                                 parse_item_cb=parse_item,
                                                 batch_size=bs, num_workers=num_threads,
                                                 shuffle=False)

    return DataProvider(item_loaders)
