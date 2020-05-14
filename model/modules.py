import torch
import torch.nn as nn
import torch.nn.functional as F

""" Attention modules """
class Attention(nn.Module):

	def __init__(self, feat_dim, hidden_dim, out_features):
		super(Attention, self).__init__()
		self.f_feat = nn.Linear(in_features=feat_dim, out_features=out_features)
		self.f_hidden = nn.Linear(in_features=hidden_dim, out_features=out_features)
		self.f_att = nn.Linear(in_features=out_features, out_features=1)
	
	def forward(self, features, hidden):
		# hidden shape after unsqueeze == [batch_size, 1, hidden_dim]
		hidden = hidden.unsqueeze(1)

		# score shape == [batch_size, 196, hidden_dim]
		score = F.relu(self.f_feat(features) + self.f_hidden(hidden)) 
		#score = torch.tanh(self.f_feat(features) + self.f_hidden(hidden)) # tanh? or relu?

		score = self.f_att(score)
		# attention weights shape == [batch_size, 196, hidden_dim]
		attention_weights = F.softmax(score, dim=1)

		return attention_weights

class Selector(nn.Module):
	def __init__(self, hidden_dim):
		super(Selector, self).__init__()
		self.fc = nn.Linear(in_features=hidden_dim, out_features=1)

	def forward(self, context_vector, hidden):
		beta = torch.sigmoid(self.fc(hidden)) # [N, hidden_dim] -> [N, 1]
		return beta * context_vector

class SoftAttention(nn.Module):

	def __init__(self, feat_dim, hidden_dim):
		super(SoftAttention, self).__init__()
		self.att = Attention(feat_dim, hidden_dim, min(feat_dim, hidden_dim))
		self.selector = Selector(hidden_dim)

	def forward(self, features, hidden):
		attention_weights = self.att(features, hidden) # [N, L, 1]

		context_vector = attention_weights * features
		context_vector = torch.mean(context_vector, dim=1) # sum or mean
		context_vector = self.selector(context_vector, hidden)
		return attention_weights, context_vector

class HardAttention(nn.Module):

	def __init__(self, feat_dim, hidden_dim):
		super(HardAttention, self).__init__()
		self.att = Attention(feat_dim, hidden_dim)

	def forward(self, a, h):
		alpha = self.att(a, h)
		raise NotImplementedError

""" LSTM modules """
class LSTMinit(nn.Module):

	def __init__(self, feat_dim, hidden_dim):
		super(LSTMinit, self).__init__()
		self.init_h = nn.Linear(in_features=feat_dim, out_features=hidden_dim)
		self.init_c = nn.Linear(in_features=feat_dim, out_features=hidden_dim)
	
	def forward(self, features):
		a_mean = torch.mean(features, dim=1)
		hidden = torch.tanh(self.init_h(a_mean))
		cell = torch.tanh(self.init_c(a_mean))
		return (hidden, cell)

""" Loss modules """
class AttentionRegularization(nn.Module):

	def __init__(self, alpha_c=0.005):
		super(AttentionRegularization, self).__init__()
		self.alpha_c = alpha_c

	def forward(self, alphas):
		alpha_reg = alphas.sum(dim=1) # [N, t, L] -> [N, L]
		alpha_reg = (1. - alpha_reg)**2
		alpha_reg = alpha_reg.sum(dim=1).mean() # [N, L] -> [N]
		return self.alpha_c * alpha_reg
	
""" Encoder & Decoder """
class Encoder(nn.Module):

	def __init__(self, feat_dim):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(in_features=feat_dim, out_features=feat_dim)

	def forward(self, x):
		x = self.fc(x)
		x = F.relu(x)
		return x

class Decoder(nn.Module):

	def __init__(self, feat_dim, hidden_dim, num_embeddings, embedding_dim, p=0.5):
		super(Decoder, self).__init__()

		self.attention = SoftAttention(feat_dim, hidden_dim)
		self.selector = Selector(hidden_dim)

		self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
		self.rnn = nn.LSTMCell(feat_dim+embedding_dim, hidden_dim) 
		self.deepout = nn.Linear(hidden_dim, num_embeddings)

		#self.layernorm = nn.LayerNorm(embedding_dim)
		self.dropout = nn.Dropout(p=p)

		self._init()

	def forward(self, token, features, state):
		hidden, cell = state
		emb = self.dropout(self.embedding(token))
		#emb = self.layernorm(emb)

		alpha, context_vector = self.attention(features, hidden)
		hidden, cell = self.rnn(torch.cat([emb, context_vector], dim=1), state)
		
		logit = self.deepout(self.dropout(hidden))
		return logit, (hidden, cell), alpha
	
	def _init(self):
		# initialize attention
		for name, param in self.attention.named_parameters():
			if 'weight' in name:
				nn.init.xavier_normal_(param.data)
			elif 'bias' in name:
				nn.init.constant_(param.data, 0.1)

		# initialize embedding
		for name, param in self.embedding.named_parameters():
			if 'weight' in name:
				nn.init.uniform_(param.data, a=-1., b=1.)
				
		# initialize lstm
		for name, param in self.rnn.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(param.data)
			elif 'bias' in name:
				nn.init.constant_(param.data, 1.)

		# initialize deep output-layer 
		for name, param in self.deepout.named_parameters():
			if 'weight' in name:
				nn.init.xavier_normal_(param.data)
			elif 'bias' in name:
				nn.init.constant_(param.data, 0.1)
