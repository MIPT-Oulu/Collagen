## load model
import torch
import numpy as np
from torchvision import datasets, transforms
from collagen.core.utils import auto_detect_device
from examples.autoencoder.models import AutoEncoder
from torch.autograd import Variable
import matplotlib.pyplot as plt


class TestData(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.asarray(datasets.MNIST('./data/', train=False, transform=None, download=True).test_data)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tmp = self.data[index]
        tmp32 = np.zeros((32, 32, 1), dtype=np.float32)
        tmp32[2:30, 2:30, 0] = tmp
        return self.transform(tmp32)


device = auto_detect_device()


model = AutoEncoder().to(device)
# replace the saved model with the file you want to test
# for better reconstruction increase the number of epochs
saved_model = 'snapshots/model_0009_20190819_144458_eval.loss_0.209.pth'
model.load_state_dict(torch.load(saved_model))


# prepare test data

test_data = TestData()

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

in_out = []
for img in test_loader:
    inp = Variable(img.cuda())
    out = model(inp)
    in_out.append([inp, out])
    r = np.random.randint(low=0, high=100)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img[r].reshape(32, 32), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(out[r].detach().cpu().numpy().reshape(32, 32), cmap='gray')
    axs[1].set_title('Reconstructed')
    plt.show()
    break


