import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from utils.data import *
from utils.functions import *
from model.modules import *

import pickle
import random

class VisualAttentionNet(pl.LightningModule):

	def __init__(self, hparams):
		super(VisualAttentionNet, self).__init__()
		self.hparams = hparams

		self.init = LSTMinit(hparams.feat_dim, hparams.hidden_dim)
		#self.enc = Encoder(hparams.feat_dim)
		self.dec = Decoder(hparams.feat_dim, hparams.hidden_dim, hparams.num_embeddings, hparams.embedding_dim, hparams.dropout)

		self.dropout = nn.Dropout(p=hparams.dropout)
		self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)
		if hparams.alpha_c > 0.:
			self.att_decay = AttentionRegularization(alpha_c=hparams.alpha_c)
		else: self.att_decay = None

		self.BStool = BeamSearchTool(hparams.num_embeddings, hparams.max_len, hparams.beam_width, hparams.dictionary_path)

		# logger
		self.optimal_bleu = 0.
		self.global_step = 0
		self.val_step = 0

	def forward(self, token, feature, state):
		logit, state, alpha = self.dec(token, feature, state)
		return logit, state, alpha

	def training_step(self, batch, batch_idx):
		feat, caption, lengths = batch
		feat = feat.transpose(1,2)
		max_len = min(max(lengths) - 1, self.hparams.max_len)
		lengths = torch.tensor(lengths).type_as(feat)
		loss = 0.

		state = self._init_lstm(feat)
		#feat = self.enc(feat)
		alphas = []
		for t in range(max_len):
			logit, state, alpha = self(caption[:,t], feat, state)
			loss += self.cross_entropy(logit, caption[:,t+1])
			alphas.append(alpha.squeeze(2).unsqueeze(1))

		loss /= max_len

		decay_term = 0.
		if self.hparams.alpha_c > 0.:
			att_decay = self.att_decay(torch.cat(alphas, dim=1))
			decay_term += att_decay
		if self.hparams.decay_c > 0.:
			weight_decay = 0.
			for name, param in self.named_parameters():
				if 'weight' in name and 'weight_' not in name:
					weight_decay += param.norm(2)
			weight_decay *= self.hparams.decay_c
			decay_term += weight_decay
		loss += decay_term
			
		self.logger.experiment.add_scalar('train/objective', loss.item(), self.global_step)
		if self.hparams.alpha_c > 0. or self.hparams.decay_c > 0.:
			self.logger.experiment.add_scalar('train/loss', loss.item()-decay_term.item(), self.global_step)
			self.logger.experiment.add_scalar('train/decay_term', decay_term.item(), self.global_step)
		self.global_step += 1

		return {'loss': loss}

	""" validation related functions """
	def validation_step(self, batch, batch_idx):
		feat, caption, imgid = batch
		feat = feat.transpose(1,2)

		# Feed forward
		state = self._init_lstm(feat)
		#feat = self.enc(feat)

		self.BStool.reset(self.dec, feat, state)
		self.BStool.step()
		sentence, alphas = self.BStool.get_sentence()

		# Evaluate
		bleu = sentence_bleu(caption[0], sentence, smoothing_function=SmoothingFunction(self.hparams.epsilon).method1)
		if 77 == random.randrange(100):
			att_img = draw_attention(imgid, sentence, alphas)
			sample = ' '.join(word for word in sentence)
			self.logger.experiment.add_image('val/attention', att_img, self.val_step, dataformats='NCHW')
		return {'bleu': bleu, 'reference':caption[0], 'hypothesis':sentence}
	
	def validation_epoch_end(self, outputs):
		references = [x['reference'] for x in outputs]
		hypotheses = [x['hypothesis'] for x in outputs]
		bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction(self.hparams.epsilon).method1)
		bleus = [x['bleu'] for x in outputs]
		avg_bleu = sum(bleus) / len(bleus)

		#self.logger.experiment.add_scalar('val/bleu-4', bleu_score, self.val_step)
		#self.logger.experiment.add_scalar('val/sentence_bleu', avg_bleu, self.val_step)
		self.val_step += 1

		if bleu_score > self.optimal_bleu:
			self.optimal_bleu = bleu_score
			torch.save(self.state_dict(), self.hparams.ckpt_path)

		logs = {'bleu-4': bleu_score, 'avg_bleu':avg_bleu}
		results = {'log': logs}
		return results

	""" test related functions """
	def test_step(self, batch, batch_idx):
		feat, caption, imgid = batch
		feat = feat.transpose(1,2)

		# Feed forward
		state = self._init_lstm(feat)
		#feat = self.enc(feat)

		self.BStool.reset(self.dec, feat, state)
		self.BStool.step()
		sentence, alphas = self.BStool.get_sentence()

		# Evaluate
		att_img = draw_attention(imgid, sentence, alphas)
		self.logger.experiment.add_image('test/attention', att_img, self.val_step, dataformats='NCHW')

		return {'reference':caption[0], 'hypothesis':sentence}
	
	def test_epoch_end(self, outputs):
		references = [x['reference'] for x in outputs]
		hypotheses = [x['hypothesis'] for x in outputs]
		bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction(epsilon=self.hparams.epsilon).method1)

		logs = {'bleu-4': bleu_score}
		results = {'log': logs}
		return results
	
	""" optimizer """
	def configure_optimizers(self):
		optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
		return optimizer

	""" data related functions """
	def prepare_data(self):
		train_tf = transforms.Compose([Reshape(), ToTensor('train')])
		test_tf = transforms.Compose([Reshape(), ToTensor()])

		self.trainset = Flickr8k(split='train', transform=train_tf)
		self.valset = Flickr8k(split='val', transform=test_tf)
		self.testset = Flickr8k(split='test', transform=test_tf)
	
	def train_dataloader(self):
		return DataLoader(self.trainset, batch_size=64, shuffle=True, num_workers=4, collate_fn=train_collate)

	def val_dataloader(self):
		return DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_collate)

	def test_dataloader(self):
		return DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_collate)

	""" helpful functions """
	def _init_lstm(self, a):
		batch_size = a.size(0)
		c0, h0 = self.init(a)
		return (c0, h0)

if __name__ == '__main__':
	from argparse import ArgumentParser
	import string
	import random
	parser = ArgumentParser()
	ckpt_path = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(4)])

	# Trainer params
	parser.add_argument('--gradient_clip_val', type=float, default=.5)
	parser.add_argument('--gpus', nargs='+', type=int, default=1)
	parser.add_argument('--max_epochs', type=int, default=40)
	parser.add_argument('--ckpt_path', type=str, default='./checkpoint/'+ckpt_path+'.ckpt')
	# Dataset params
	parser.add_argument('--data_path', type=str, default=None)
	parser.add_argument('--dictionary_path', type=str, default='data/flickr8k_dictionary.pkl')
	# Model params
	parser.add_argument('--feat_dim', type=int, default=512)
	parser.add_argument('--hidden_dim', type=int, default=1024)
	parser.add_argument('--num_embeddings', type=int, default=8000)
	parser.add_argument('--embedding_dim', type=int, default=512)
	# Beam search params
	parser.add_argument('--max_len', type=int, default=50)
	parser.add_argument('--beam_width', type=int, default=9)
	parser.add_argument('--epsilon', type=float, default=1e-6)
	# Regularization params
	parser.add_argument('--dropout', type=float, default=.5)
	parser.add_argument('--decay_c', type=float, default=0.0)
	parser.add_argument('--alpha_c', type=float, default=1.0) # higher -> better?
	# optimization params
	parser.add_argument('--optimizer', type=str, default='RMSprop')
	parser.add_argument('--lr', type=float, default=1e-3)

	hparams = parser.parse_args()
	hparams.gpus=[3]

	print(hparams)
	trainer = pl.Trainer().from_argparse_args(hparams, checkpoint_callback=False)
	model = VisualAttentionNet(hparams)
	trainer.fit(model)
