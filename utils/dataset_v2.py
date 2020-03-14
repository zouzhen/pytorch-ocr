from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import torch
from skimage.transform import resize
from PIL import Image

class baiduDataset(Dataset):
	def __init__(self, img_root, label_path, alphabet, isBaidu, params, transforms=None):
		super(baiduDataset, self).__init__()
		self.img_root = img_root
		self.isBaidu = isBaidu
		self.labels = self.get_labels(label_path)
		self.alphabet = alphabet
		self.transforms = transforms
		self.params = params
		self.width, self.height = params.imgW, params.imgH
	def get_labels(self, label_path):
		# return text labels in a list
		if self.isBaidu:
			with open(label_path, 'r', encoding='utf-8') as file:
				content = [[{c.split('\t')[2]:c.split('\t')[3][:-1]},{"w":c.split('\t')[0]}] for c in file.readlines()];
			labels = [c[0] for c in content]
		else:
			with open(label_path, 'r', encoding='utf-8') as file:
				labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in file.readlines()]	
		return labels


	def __len__(self):
		return len(self.labels)

	def preprocessing(self, image):

		## already have been computed
		image = image.astype(np.float32) / 255.
		image = torch.from_numpy(image).type(torch.FloatTensor)
		image.sub_(self.params.mean).div_(self.params.std)

		return image

	def __getitem__(self, index):
		image_name = list(self.labels[index].keys())[0]
		# label = list(self.labels[index].values())[0]
		# print(self.img_root+'/'+image_name)
		image = cv2.imread(self.img_root+'/'+image_name)
		# print(self.img_root+'/'+image_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		h, w = image.shape
		image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
		image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
		image = self.preprocessing(image)

		return image, index

		


if __name__ == '__main__':
	dataset = baiduDataset("H:/DL-DATASET/BaiduTextR/train_images/train_images", "H:/DL-DATASET/BaiduTextR/train.list", params.alphabet, True)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
	
	for i_batch, (image, index) in enumerate(dataloader):
		print(image.shape)
