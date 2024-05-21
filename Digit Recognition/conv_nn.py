import os
import torch
from torch.nn import Module
from torch.nn import Flatten, Sequential, Conv2d, MaxPool2d, Linear, ReLU, Softmax, CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam

import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

train_data = MNIST(root="dataset", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="dataset", train=False, download=True, transform=ToTensor())

print(train_data.data.shape)
plt.imshow(train_data.data[222].numpy(), cmap="gray")
plt.show()
# exit()


num_train_samples = int(len(train_data)*TRAIN_SPLIT)
num_val_samples = int(len(train_data)*VAL_SPLIT)

(train_data, val_data) = random_split(train_data,
									  lengths = [num_train_samples, num_val_samples],
									  generator=torch.Generator().manual_seed(1))

train_data_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

train_steps = len(train_data_loader.dataset)
val_steps = len(val_data_loader.dataset)

class Inet(Module):
	def __init__(self, num_channels, num_classes):
		super().__init__()
		self.net = Sequential(
			Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5)),
			ReLU(),
			MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
			ReLU(),
			MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			Flatten(start_dim=1),
			Linear(in_features=800, out_features=500),
			ReLU(),
			Linear(in_features=500, out_features=num_classes),
			Softmax(dim=0)
		)

	def forward(self, x):
		out = self.net(x)
		return out


model = Inet(num_channels=1, num_classes=10).to(device)
opt = Adam(model.parameters(), lr=INIT_LR)
lossFN = CrossEntropyLoss()
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

print(model)

for e in range(EPOCHS):
	model.train()
	total_train_loss = 0
	total_val_loss = 0
	train_correct = 0
	val_correct = 0

	for (x, y) in train_data_loader:
		(x, y) = (x.to(device), y.to(device))

		pred = model(x)
		loss = lossFN(pred, y)

		opt.zero_grad()
		loss.backward()
		opt.step()

		total_train_loss+=loss
		train_correct+=(pred.argmax(1)==y).type(torch.float64).sum().item()

	with torch.no_grad():
		model.eval()

		for (x, y) in val_data_loader:
			(x, y) = (x.to(device), y.to(device))
			pred = model(x)
			total_val_loss +=lossFN(pred, y)

			val_correct += (pred.argmax(1)==y).type(torch.float64).sum().item()

	avg_train_loss = total_train_loss/train_steps
	avg_val_loss = total_val_loss/val_steps

	train_correct = train_correct/len(train_data_loader.dataset)
	val_correct = val_correct/len(val_data_loader.dataset)

	H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
	H["train_acc"].append(train_correct)
	H["val_loss"].append(avg_val_loss.cpu().detach().numpy())
	H["val_acc"].append(val_correct)

	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avg_train_loss, train_correct))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avg_val_loss, val_correct))

with torch.no_grad():
	model.eval()
	preds = []

	for (x, y) in test_data_loader:
		x = x.to(device)
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

print(classification_report(test_data.targets.cpu().numpy(),
							np.array(preds), target_names=test_data.classes))


path = os.path.join("dataset", "MNIST", "state_dict")
os.makedirs(path, exist_ok=True)

torch.save(model.state_dict(), os.path.join(path, 'model_weights.pth'))


