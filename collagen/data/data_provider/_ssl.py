from collagen.data import ItemLoader, AugmentedGroupSampler, AugmentedGroupStudentTeacherSampler, DataProvider


def pimodel_data_provider(model, train_labeled_data, train_unlabeled_data, val_labeled_data, val_unlabeled_data,
                          transforms, parse_item, bs, num_threads, item_loaders=dict(), root=""):
    """
    Default setting of data provider for Pi-Model

    """
    item_loaders["labeled_train"] = ItemLoader(root=root, meta_data=train_labeled_data,
                                               transform=transforms[0],
                                               parse_item_cb=parse_item,
                                               batch_size=bs, num_workers=num_threads,
                                               shuffle=True)

    item_loaders["unlabeled_train"] = AugmentedGroupSampler(root=root, model=model,
                                                            meta_data=train_unlabeled_data,
                                                            n_augmentations=1,
                                                            augmentation=transforms[2],
                                                            transform=transforms[1],
                                                            parse_item_cb=parse_item,
                                                            batch_size=bs, num_workers=num_threads,
                                                            shuffle=True)

    item_loaders["labeled_eval"] = ItemLoader(root=root, meta_data=val_labeled_data,
                                              transform=transforms[1],
                                              parse_item_cb=parse_item,
                                              batch_size=bs, num_workers=num_threads,
                                              shuffle=False)

    item_loaders["unlabeled_eval"] = AugmentedGroupSampler(root=root, model=model,
                                                           meta_data=val_unlabeled_data,
                                                           n_augmentations=1,
                                                           augmentation=transforms[2],
                                                           transform=transforms[1],
                                                           parse_item_cb=parse_item,
                                                           batch_size=bs, num_workers=num_threads,
                                                           shuffle=False)

    return DataProvider(item_loaders)


def mt_data_provider(st_model, te_model, train_labeled_data, train_unlabeled_data, val_labeled_data,
                     val_unlabeled_data,
                     transforms, parse_item, bs, num_threads, item_loaders=dict(), root=""):
    """
    Default setting of data provider for Mean-Teacher

    """
    item_loaders["labeled_train"] = AugmentedGroupStudentTeacherSampler(root=root, name='l',
                                                                        meta_data=train_labeled_data,
                                                                        student_model=st_model,
                                                                        teacher_model=te_model,
                                                                        n_augmentations=1,
                                                                        augmentation=transforms[2],
                                                                        transform=transforms[1],
                                                                        parse_item_cb=parse_item,
                                                                        batch_size=bs, num_workers=num_threads,
                                                                        shuffle=True)

    item_loaders["unlabeled_train"] = AugmentedGroupStudentTeacherSampler(root=root, name='u',
                                                                          student_model=st_model,
                                                                          teacher_model=te_model,
                                                                          meta_data=train_unlabeled_data,
                                                                          n_augmentations=1,
                                                                          augmentation=transforms[2],
                                                                          transform=transforms[1],
                                                                          parse_item_cb=parse_item,
                                                                          batch_size=bs, num_workers=num_threads,
                                                                          shuffle=True)

    item_loaders["labeled_eval"] = AugmentedGroupStudentTeacherSampler(root=root, name='l',
                                                                       meta_data=val_labeled_data,
                                                                       student_model=st_model,
                                                                       teacher_model=te_model,
                                                                       n_augmentations=1,
                                                                       augmentation=transforms[2],
                                                                       transform=transforms[1],
                                                                       parse_item_cb=parse_item,
                                                                       batch_size=bs, num_workers=num_threads,
                                                                       shuffle=False)

    item_loaders["unlabeled_eval"] = AugmentedGroupStudentTeacherSampler(root=root, name='u',
                                                                         student_model=st_model,
                                                                         teacher_model=te_model,
                                                                         meta_data=val_unlabeled_data,
                                                                         n_augmentations=1,
                                                                         augmentation=transforms[2],
                                                                         transform=transforms[1],
                                                                         parse_item_cb=parse_item,
                                                                         batch_size=bs, num_workers=num_threads,
                                                                         shuffle=False)

    return DataProvider(item_loaders)