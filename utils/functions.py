import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import os
import io

class BeamSearchTool:

	def __init__(self, vocab_size, max_len, beam_width, dictionary_path):
		self.B = beam_width
		self.V = vocab_size
		self.max_len = max_len
		self.norm_factor = 1. / torch.pow(torch.tensor(max_len, dtype=torch.float), 0.7)
		self.vocab_to_idx, self.idx_to_vocab = pickle.load(open(dictionary_path,'rb'))
	
	def step(self):
		for _ in range(self.max_len):
			scores, states, atts = [], [], []
			for i in range(self.B):
				_logit, _state, _att = self.model(self.candidates[i]['sentence'][-1], self.feat, self.candidates[i]['state'])
				_score = F.log_softmax(_logit, dim=1) + self.candidates[i]['prob']
				scores.append(_score)
				states.append(_state)
				atts.append(_att)
			scores = torch.cat(scores, dim=0)
			scores = scores.view(1, -1)
			top_score, top_idx = scores.topk(k=self.B)
			b, words = top_idx[0]//self.V, top_idx[0]%self.V
			top_score = top_score.squeeze(0)

			new_candidates = []
			for j in range(self.B):
				prev, word = b[j].item(), words[j].item()
				_sentence = self.candidates[prev]['sentence'].copy()
				_sentence.append(words[j].unsqueeze(0))
				_state = states[prev]
				_prob = top_score[j]
				_alphas = self.candidates[prev]['atts'].copy()
				_alphas.append(atts[prev])
				candidate = {'sentence':_sentence, 'state':_state, 'prob': _prob, 'atts':_alphas}
				new_candidates.append(candidate)

			self.candidates = new_candidates

	def get_sentence(self):
		max_p = -1e9
		argmax = -1
		sentences = []
		for i in range(self.B):
			if self.candidates[i]['prob'] > max_p:
				max_p = self.candidates[i]['prob']
				argmax = i

		sentence = []
		for token in self.candidates[argmax]['sentence']:
			word = self.idx_to_vocab[token.item()]
			sentence.append(word)
			if word == '.':
				break
		return sentence, self.candidates[argmax]['atts']
	
	def reset(self, model, feat, state):
		self.model = model
		self.feat = feat
		self.candidates = []

		start_token = torch.tensor(self.vocab_to_idx['<START>']).unsqueeze(0).to(feat.device)
		_logit, _state, _att = self.model(start_token, feat, state)
		_score = F.log_softmax(_logit, dim=1)
		_alphas = [_att]

		top_score, top_idx = _score.topk(k=self.B)
		for i in range(self.B):
			candidate = {'sentence':[top_idx.squeeze()[i].unsqueeze(0)], 'state':_state, 'prob': top_score.squeeze()[i], 'atts':_alphas}
			self.candidates.append(candidate)

def draw_attention(img_id, sentence, attention_weights, img_list='data/flickr8k_image_list.pkl', root_dir='/mnt/sdb/Flickr8k/Flicker8k_Dataset/'):
	"""
	https://www.tensorflow.org/tutorials/text/image_captioning
	"""
	length = len(sentence)
	img_list = pickle.load(open(img_list, 'rb'))

	# load image
	img_path = os.path.join(root_dir, img_list[img_id])
	ori_img = Image.open(img_path)

	# draw images
	fig = plt.figure(figsize=(8,8))
	nrows, ncols = (length+1)//4 + min((length+1)%4,1), 4

	ax = fig.add_subplot(nrows, ncols, 1)
	img = ax.imshow(ori_img)
	ax.axis('off')
	for i in range(length):
		ax = fig.add_subplot(nrows, ncols, i+2)
		img = ax.imshow(ori_img)
		att = attention_weights[i-1].view(14,14).cpu().numpy()
		ax.imshow(att, cmap='gray', alpha=.6, extent=img.get_extent())
		ax.axis('off')
		ax.set_title(sentence[i])

	buf = io.BytesIO()
	fig.tight_layout()
	fig.savefig(buf, format='jpeg')
	buf.seek(0)
	plt.close('all')

	image = Image.open(buf)
	image = transforms.ToTensor()(image).unsqueeze(0)
	return image
