import pytest
import pandas as pd 
import numpy as np
import torch 


@pytest.fixture(scope='module', params=[32, 500])
def metadata_fname_target_5_classes(request):
    np.random.seed(42)
    return pd.DataFrame(data={'fname': [f'img_{i}.png' for i in range(request.param)],
                              'target': np.random.choice(5, request.param, replace=True)})


@pytest.fixture
def ones_image_parser():
    def parser(data_dir, entry, transforms):
        img = np.ones((9, 9))
        target = entry.target

        img, target = transforms(img, target)

        return {'img': img, 'target': target, 'fname': entry.fname}
    return parser


@pytest.fixture
def img_target_transformer():
    def empty_transform(img, target):
        return torch.from_numpy(img.astype(np.float32)).unsqueeze(0), target
    return empty_transform
