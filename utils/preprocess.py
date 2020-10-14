import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models

from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import json
import os
from collections import Counter

gpu = torch.cuda.is_available()
transform = transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.4640, 0.4443, 0.4021], std=[0.2720, 0.2636, 0.2757])
	])

class FlickrImage(Dataset):

	def __init__(self, filenames, root_dir, transform=None):
		self.filenames = filenames
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		filepath = os.path.join(self.root_dir, self.filenames[idx])
		image = Image.open(filepath)

		if self.transform:
			image = self.transform(image)
		return image

class FeatureExtractor(nn.Module):

	def __init__(self):
		super(FeatureExtractor, self).__init__()
		vgg = models.vgg19(pretrained=True)
		print(vgg)
		self.encoder = nn.Sequential(*list(vgg.features.children())[:-1])
		print(self.encoder)
		
	def forward(self, x):
		return self.encoder(x)

def create_pkl(dataset, json_path, image_path, max_words, save_path, extract):
	""" 
	Creates pickle files

	dataset		: dataset to create ['coco', 'flickr8k', 'flickr30k']
	json_path	: path of JSON file created by Andrej Karpath https://cs.stanford.edu/people/karpathy/deepimagesent/
	image_path	: path of image folder
	save_path	: path where pickle file will be saved
	"""

	assert dataset in {'coco', 'flickr8k', 'flickr30k'}

	# Read json file
	with open(json_path, 'r') as f:
		json_file = json.load(f)['images']

	# Read json file
	captions = []
	vocabs = []
	filenames = []
	for data in tqdm(json_file, desc='Reading json file ...'):
		filenames.append(data['filename'])
		for sentence in data['sentences']:
			captions.append(sentence['tokens'])
			for vocab in sentence['tokens']:
				vocabs.append(vocab)
	
	# Create dictionary
	print("Creating dictionary ...")
	dictionary = [word for word, value in Counter(vocabs).most_common()]
	dictionary = dictionary[:max_words-4]
	dictionary.insert(0, '<PAD>')
	dictionary.append('<START>')
	dictionary.append('.')
	dictionary.append('<UNK>')
	vocab_to_idx = {vocab:i for i, vocab in enumerate(dictionary)}
	idx_to_vocab = {i:vocab for i, vocab in enumerate(dictionary)}
	dictionary = (vocab_to_idx, idx_to_vocab)
	print('Total vocabulary size: ', len(vocab_to_idx))

	# Convert vocab to int and extract features
	dataset = {'train':[], 'val':[], 'test':[]}
	for data in tqdm(json_file, desc='Converting vocab to int ...'):
		split = data['split']
		converted = []
		if split == 'train':
			for i in range(5):
				converted = []
				for vocab in data['sentences'][i]['tokens']:
					try:
						converted.append(vocab_to_idx[vocab])
					except:
						converted.append(vocab_to_idx['<UNK>'])
				converted.insert(0, vocab_to_idx['<START>'])
				converted.append(vocab_to_idx['.'])
				dataset[split].append({'imgid':data['imgid'], 'caption':converted})
		else:
			for sentence in data['sentences']:
				tmp = sentence['tokens']
				tmp.append('.')
				converted.append(tmp)
			dataset[split].append({'imgid':data['imgid'], 'caption':converted})

	# Extract features from image
	feats = None
	if extract is True:
		feats = []
		extractor = FeatureExtractor().eval()
		img_dataset = FlickrImage(filenames, image_path, transform)
		img_loader = DataLoader(img_dataset, batch_size=200, shuffle=False)
		if gpu:
			extractor.cuda()
		with torch.no_grad():
			for img in tqdm(img_loader, desc='Extracting feature ...'):
				if gpu:
					img = img.cuda()
				feat = extractor(img).cpu()
				feats.append(feat)
		feats = torch.cat(feats, dim=0).numpy()

	# Save files
	data_path = save_path + '_image_list.pkl'
	print('Saved image list at ', data_path)
	with open(data_path, 'wb') as f:
		pickle.dump(filenames, f)

	data_path = save_path + '_dictionary.pkl'
	print('Saved dictionary at ', data_path)
	with open(data_path, 'wb') as f:
		pickle.dump(dictionary, f)

	data_path = save_path + '_captions.pkl'
	print('Saved dataset at ', data_path)
	with open(data_path, 'wb') as f:
		pickle.dump(dataset, f)

	if feats is not None:
		feat_path = save_path + '_feats.pkl'
		print('Saved extracted features at ', feat_path)
		with open(feat_path, 'wb') as f:
			pickle.dump(feats, f)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='flickr8k')
	parser.add_argument('--json_path', type=str, default='data/dataset_flickr8k.json')
	parser.add_argument('--img_path', type=str, default='/mnt/sdb/Flickr8k/Flicker8k_Dataset')
	parser.add_argument('--max_words', type=int, default=8000)
	parser.add_argument('--extract', type=bool, default=False)
	parser.add_argument('--save_path', type=str, default='data/flickr8k')
	args = parser.parse_args()

	# Preprocess Andrej Karpathy's json file
	create_pkl(args.dataset, args.json_path, args.img_path, args.max_words, args.save_path, args.extract)
