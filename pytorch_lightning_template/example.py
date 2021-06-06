import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Compose

from pytorch_lightning_template import LightningModuleTemplate

print(pl.__version__)
print(torch.__version__)

train_transform = Compose([
    ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    Normalize(mean=(0.5,), std=(0.5,))
])
val_transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5,), std=(0.5,))
])

train_data = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_transform)
validation_data = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=val_transform)

batch_size = bs = 1000
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=4)
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=bs, num_workers=4)

x_0, y_0 = train_data[0]

# Hyper parameters
epochs = 200
lr = 2e-3
n = 64


class MnistClassifierCNN(LightningModuleTemplate):
    def __init__(self, n, input_shape, acc_fn, loss_fn, opt_fn, lr=1e-4):
        super().__init__()
        self.epoch = 0
        self.input_shape = input_shape
        self.acc_fn = acc_fn
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.lr = lr
        self.metrics = {}

        self.num_of_classes = 10
        self.input_shape = input_shape

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(1)

        self.conv1 = nn.Conv2d(input_shape[0], n, 3)
        self.conv2 = nn.Conv2d(n, n, 3, 2)
        self.conv3 = nn.Conv2d(n, 2 * n, 3)
        self.conv4 = nn.Conv2d(2 * n, 2 * n, 3, 2)
        self.bn = nn.BatchNorm1d(25 * 2 * n)
        self.out = nn.Linear(25 * 2 * n, self.num_of_classes)

    def forward(self, input_x):
        x = input_x
        # out = self.seq(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.bn(x)
        x = self.out(x)
        out = self.softmax(x)
        return out


def acc(y_hat, y):
    return float(torch.where(torch.argmax(y_hat, axis=1) == y)[0].shape[0]) / float(y_hat.shape[0])


model = MnistClassifierCNN(
    n=n,
    input_shape=x_0.shape,
    lr=lr,
    acc_fn=acc,
    loss_fn=torch.nn.CrossEntropyLoss(),
    opt_fn=torch.optim.Adam
)
model.summary()

trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=[validation_dataloader])
