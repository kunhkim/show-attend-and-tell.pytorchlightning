import torch
import pytorch_lightning as pl

from train import VisualAttentionNet

if __name__ == '__main__':
	from argparse import Namespace, ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--ckpt_path', type=str, default='./checkpoint/XOoy_22.ckpt')
	args = parser.parse_args()

	# Dataset params
	hparams = Namespace(
		data_path=None,
		dictionary_path='data/flickr8k_dictionary.pkl',
		feat_dim=512,
		hidden_dim=1024,
		num_embeddings=8000,
		embedding_dim=512,
		max_len=50,
		beam_width=9,
		epsilon=1e-6,
		dropout=.0,
		decay_c=0,
		alpha_c=0,
	)
		
	print(hparams)
	trainer = pl.Trainer().from_argparse_args(hparams, checkpoint_callback=False)
	model = VisualAttentionNet(hparams)
	model.load_state_dict(torch.load(args.ckpt_path)) # g28o_12.ckpt
	model.prepare_data()
	trainer.test(model)
