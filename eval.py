import torch
import pytorch_lightning as pl

from train import VisualAttentionNet

if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()

	# Trainer params
	parser.add_argument('--gradient_clip_val', type=float, default=.5)
	parser.add_argument('--gpus', nargs='+', type=int, default=0)
	parser.add_argument('--max_epochs', type=int, default=40)
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
	hparams.gpus=[0]

	print(hparams)
	trainer = pl.Trainer().from_argparse_args(hparams, checkpoint_callback=False)
	model = VisualAttentionNet(hparams)
	model.load_state_dict(torch.load('./checkpoint/XOoy_22.ckpt')) # g28o_12.ckpt
	model.prepare_data()
	trainer.test(model)
