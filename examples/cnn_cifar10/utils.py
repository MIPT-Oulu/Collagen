import solt
import solt.transforms as slt


def parse_item(root, entry, trf, data_key, target_key):
    img = entry[data_key]

    stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.243, 0.261)}

    trf_data = trf({'image': img}, normalize=True, **stats)
    return {data_key: trf_data['image'].half(), target_key: entry[target_key]}


def my_transforms():
    train_trf = solt.Stream([
        slt.Pad(pad_to=(36, 36)),
        slt.Rotate(10),
        slt.Crop((32, 32)),
        slt.CutOut((8, 8)),
        slt.Flip(p=0.5)
    ])

    test_trf = solt.Stream([])

    return {'train': train_trf, 'eval': test_trf}
