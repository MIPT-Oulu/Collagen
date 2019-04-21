from collagen.data import ItemLoader
import itertools

import pytest
from .fixtures import metadata_fname_target_5_classes, ones_image_parser, img_target_transformer


@pytest.mark.parametrize('batch_size, n_samples', itertools.product([32, 11, 3], [1, 3, 6]))
def test_loader_samples_batches(batch_size, n_samples, metadata_fname_target_5_classes,
                                ones_image_parser, img_target_transformer):

    iterm_loader = ItemLoader(meta_data=metadata_fname_target_5_classes, root='/tmp/',
                              batch_size=batch_size, parse_item_cb=ones_image_parser,
                              transform=img_target_transformer, shuffle=True)

    samples = iterm_loader.sample(n_samples)

    assert len(samples) == n_samples
    assert samples[0]['img'].size(0) == batch_size
    assert samples[0]['target'].size(0) == batch_size


@pytest.mark.parametrize('batch_size, n_samples', itertools.product([8], [5, 25]))
def test_loader_endless_sampling_works(batch_size, n_samples, metadata_fname_target_5_classes,
                                       ones_image_parser, img_target_transformer):

    iterm_loader = ItemLoader(meta_data=metadata_fname_target_5_classes, root='/tmp/',
                              batch_size=batch_size, parse_item_cb=ones_image_parser,
                              transform=img_target_transformer, shuffle=True)

    for i in range(2*len(iterm_loader)):
        samples = iterm_loader.sample(n_samples)

        assert len(samples) == n_samples
        assert samples[0]['img'].size(0) == batch_size
        assert samples[0]['target'].size(0) == batch_size


@pytest.mark.parametrize('batch_size, n_samples, drop_last', itertools.product([3, 32], [1, 2], [True, False]))
def test_loader_drop_last(batch_size, n_samples, metadata_fname_target_5_classes,
                          ones_image_parser, img_target_transformer, drop_last):

    iterm_loader = ItemLoader(meta_data=metadata_fname_target_5_classes, root='/tmp/',
                              batch_size=batch_size, parse_item_cb=ones_image_parser,
                              transform=img_target_transformer, shuffle=True, drop_last=drop_last)

    if drop_last:
        assert len(iterm_loader) == metadata_fname_target_5_classes.shape[0] // batch_size
    else:
        if metadata_fname_target_5_classes.shape[0] % batch_size != 0:
            assert len(iterm_loader) == metadata_fname_target_5_classes.shape[0] // batch_size + 1
        else:
            assert len(iterm_loader) == metadata_fname_target_5_classes.shape[0] // batch_size

