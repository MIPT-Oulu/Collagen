import pandas as pd
import os
from torchvision.datasets import MNIST
from torchvision import transforms
# from torchvision.datasets import ImageFolder

def standardize_imagefolder(input_dir: str, extensions: list = ["png", "jpg"]):
    list_rows = []

    classes = []
    for cls in os.listdir(input_dir):
        sub_dir = os.path.join(input_dir, cls)
        if os.path.isdir(sub_dir) or os.path.islink(sub_dir):
            classes.append(cls)
            # print('Class: {}'.format(cls))
            for root, dirs, files in os.walk(sub_dir):
                for file_name in files:
                    if file_name[-3:] in extensions:
                        file_fullname = os.path.join(root, file_name)
                        file_relname = file_fullname[len(sub_dir):]
                        row = {"img_name": file_relname, "class": cls}
                        list_rows.append(row)

    classes = list(set(classes)).sort()
    meta_data = pd.DataFrame(list_rows)
    return meta_data, classes

def standardize_mnist():
    mnist_db = MNIST(".", train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]), download=True)
    list_rows = [{"img_data": item, "label": target} for item, target in mnist_db]
    meta_data = pd.DataFrame(list_rows)

    return meta_data, list(range(10))


def standardize_csv(csv_fullname: str, column_names: list, sep: str or None = "|"):
    return pd.read_csv(csv_fullname, sep=sep, names=column_names)
