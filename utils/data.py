import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pickle
import os



""" Dataset """
class Flickr8k(Dataset):

	def  __init__(self, token_file='flickr8k_captions.pkl', feat_file='flickr8k_feats.pkl', root_dir='data/', split='train', transform=None):
		token_path = os.path.join(root_dir, token_file)
		feat_path = os.path.join(root_dir, feat_file)

		with open(feat_path, 'rb') as f:
			self.feats = pickle.load(f)
		with open(token_path, 'rb') as f:
			self.captions = pickle.load(f)[split]

		self.split = split
		self.transform = transform
	
	def __len__(self):
		return len(self.captions)

	def __getitem__(self, idx):
		feat = self.feats[self.captions[idx]['imgid']]
		caption = self.captions[idx]['caption']
		sample = (feat, caption)
		if self.transform:
			sample = self.transform(sample)

		if self.split == 'train':
			return sample
		else: 
			return sample, self.captions[idx]['imgid']

""" Transforms """
class Reshape(object):

	def __init__(self):
		pass
	
	def __call__(self, sample):
		feat, caption = sample
		d, w, h = feat.shape
		feat = feat.reshape((d, w*h))
		return (feat, caption)

class ToTensor(object):
	
	def __init__(self, split=None):
		self.split = split
	
	def __call__(self, sample):
		feat, caption = sample
		feat = torch.tensor(feat)
		if self.split is 'train':
			caption = torch.tensor(caption)
		return (feat, caption)

""" Collate functions """
def train_collate(batch):
	feats = []
	captions = []
	lengths = []
	for feat, caption in batch:
		feats.append(feat.unsqueeze(0))
		captions.append(caption)
		lengths.append(len(caption))
	feats = torch.cat(feats)
	padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
	return (feats, padded, lengths)

def test_collate(batch):
	feats = []
	captions = []
	for (feat, caption), imgid in batch:
		feats.append(feat.unsqueeze(0))
		captions.append(caption)
	feats = torch.cat(feats)
	return (feats, captions, imgid)
