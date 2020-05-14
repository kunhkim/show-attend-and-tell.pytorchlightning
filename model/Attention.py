import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

	def __init__(self, feat_dim, hidden_dim, out_features):
		super(Attention, self).__init__()
		self.f_a = nn.Linear(in_features=feat_dim, out_features=out_features)
		self.f_h = nn.Linear(in_features=hidden_dim, out_features=out_features)
		self.f_att = nn.Linear(in_features=out_features, out_features=1)

		self.feat_dim = feat_dim
		self.hidden_dim = hidden_dim
	
	def forward(self, a, h):
		batch_size = a.size(0)
		a = a.transpose(1,2) # [N, L, D]
		num_feats = a.size(1)
		h = h.unsqueeze(1).repeat(1, num_feats, 1) # [N, L, hidden_dim]

		att_a = self.f_a(a.reshape(batch_size*num_feats, self.feat_dim))
		att_h = self.f_h(h.reshape(batch_size*num_feats, self.hidden_dim))
		score = F.relu(att_a + att_h) # tanh? or relu?
		score = self.f_att(score)
		score = score.view(batch_size, -1)

		alpha = F.softmax(score, dim=1)
		return alpha

class SoftAttention(nn.Module):

	def __init__(self, feat_dim, hidden_dim):
		super(SoftAttention, self).__init__()
		self.att = Attention(feat_dim, hidden_dim, min(feat_dim, hidden_dim))

	def forward(self, a, h):
		alpha = self.att(a, h) # [N, L]
		#a = a.transpose(1,2) # [N, L, D] -> [N, D, L]
		z = a @ alpha.unsqueeze(2)
		return alpha, z.squeeze(2)

class HardAttention(nn.Module):

	def __init__(self, feat_dim, hidden_dim):
		super(HardAttention, self).__init__()
		self.att = Attention(feat_dim, hidden_dim)

	def forward(self, a, h):
		alpha = self.att(a, h)
		raise NotImplementedError

