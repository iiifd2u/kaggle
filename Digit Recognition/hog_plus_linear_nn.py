import os
import numpy as np
import cv2
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.nn import Module
from torch.nn import Linear, Softmax, Dropout, CrossEntropyLoss, Sequential
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Работаем на {device}")

transforms = Compose([ToTensor(), Lambda(lambd=lambda x:hog(x, 7, 2))])

learning_rate = 1e-3
batch_size = 64
epochs = 10
train_split = 0.75
val_split = 1 - train_split

train_data = MNIST(root="dataset", train=True, download=True, transform=transforms)
test_data = MNIST(root="dataset", train=False, download=True, transform=transforms)

num_train_samples = int(len(train_data)*train_split)
num_val_samples = int(len(train_data)*val_split)

(train_data, val_data) = random_split(train_data,
									  lengths = [num_train_samples, num_val_samples],
									  generator=torch.Generator().manual_seed(1))

train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data_loader = DataLoader(val_data, batch_size=batch_size)
test_data_loader = DataLoader(test_data, batch_size=batch_size)

train_steps = len(train_data_loader.dataset)
val_steps = len(val_data_loader.dataset)



class OneLayer(Module):
	def __init__(self, n_classes:int):
		super().__init__()
		self.net = Sequential(
			Linear(in_features=324, out_features=n_classes),
			Dropout(0.2),
			Softmax()
		)

	def forward(self, x):
		out = self.net(x)
		return out


def hog(img: torch.Tensor, block_size: int, norm_size: int):

	img = img.to(torch.float32).numpy().astype(np.float32).reshape((28, 28))
	img = img / 225.0
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

	angle = np.where(angle>=180, angle-180, angle)
	w, h = img.shape

	step_x = w//(block_size)-1
	step_y = h // (block_size)-1

	#Едет нормировочный блок 2хх
	output = np.array([])
	for i_norm in range(step_x):
		for j_norm in range(step_y):

			# Внутри него едет блок 1х1
			vector = []
			for num_x in range(norm_size):
				for num_y in range(norm_size):

					histogram = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
					# внутри блока берём координаты

					for i in range(block_size):
						for j in range(block_size):
							el = angle[i_norm*block_size+num_x*norm_size+i][j_norm*block_size+num_y*norm_size+j]
							hist_idx = el//20
							hist_idx_next = (hist_idx+1)%9
							coef = el/20 - hist_idx
							histogram[hist_idx]+= el*coef
							histogram[hist_idx_next] += el * (1-coef)
					vector+=histogram.values()
			vector = np.array(vector) / (np.linalg.norm(vector, 2)+0.001)
			output = np.concatenate([output, vector], axis=-1)

	return torch.from_numpy(output).to(torch.float32)

if __name__ == '__main__':

	model = OneLayer(10).to(device)
	opt = Adam(model.parameters(), lr=learning_rate)

	print(model)

	for e in range(epochs):
		model.train()
		total_train_loss = 0
		total_val_loss = 0
		train_correct = 0
		val_correct = 0

		for (x, y) in train_data_loader:
			(x, y) = (x.to(device), y.to(device))

			pred = model(x)
			loss = CrossEntropyLoss()(pred, y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			total_train_loss += loss
			train_correct += (pred.argmax(1) == y).type(torch.float64).sum().item()

		with torch.no_grad():
			model.eval()

			for (x, y) in val_data_loader:
				(x, y) = (x.to(device), y.to(device))
				pred = model(x)
				total_val_loss += CrossEntropyLoss(pred, y)

				val_correct += (pred.argmax(1) == y).type(torch.float64).sum().item()

		avg_train_loss = total_train_loss / train_steps
		avg_val_loss = total_val_loss / val_steps

		train_correct = train_correct / len(train_data_loader.dataset)
		val_correct = val_correct / len(val_data_loader.dataset)

		print("EPOCH: {}/{}".format(e + 1, epochs))
		print("Train loss: {:.4f}, Train accuracy: {:.4f}".format(
			avg_train_loss, train_correct))
		print("Val loss: {:.4f}, Val accuracy: {:.4f}\n".format(
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



	# pred_prob = model(output)
	# print(pred_prob)
	# y_pred = pred_prob.argmax()
	# print(y_pred)
