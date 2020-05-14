import torch
import torch.nn as nn

from model.modules import * 
from model.Attention import *

class Decoder(nn.Module):
	
	def __init__(self, feat_dim, hidden_dim,
					num_embeddings, embedding_dim, dropout=.5):
