import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt
from model.charbilstm import CharBiLSTM

class CNNLSTM(nn.Module):
	def __init__(self, vocab, num_classes, char_alphabet):

		super(CNNLSTM, self).__init__()
		self.embedding = vocab.init_embed_layer()
		self.hidden_size = opt.hidden_size

		# charcnn
		self.char_hidden_dim = 10
		self.char_embedding_dim = 20
		self.char_feature = CharBiLSTM(len(char_alphabet), None, self.char_embedding_dim, self.char_hidden_dim,
									opt.dropout, opt.gpu)

		D = self.embedding.weight.size(1)
		self.hidden_size = opt.hidden_size
		D = D + self.char_hidden_dim

		#mention cnn
		Ci = 1
		Co = opt.kernel_num
		Ks = [int(k) for k in list(opt.kernel_sizes) if k != ","]
		self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
											   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])


		self.hidden = nn.Linear(len(Ks) * Co, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, num_classes)
		self.dropout = nn.Dropout(opt.dropout)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, x, char_inputs):
		inputs, lengths, seq_recover = x
		x = self.embedding(inputs)# (N, W, D)

		batch_size, max_len = inputs.size()
		char_inputs, char_seq_lengths, char_seq_recover = char_inputs
		char_features = self.char_feature.get_last_hiddens(char_inputs,char_seq_lengths.cpu().numpy())
		char_features = char_features[char_seq_recover]
		char_features = char_features.view(batch_size, max_len, -1)

		x = torch.cat((x, char_features), 2)

		x = x.unsqueeze(1)  # (N, Ci, W, D)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
		x = torch.cat(x, 1)

		'''
		x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
		x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
		x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
		x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
		'''
		x = self.dropout(x)  # (N, len(Ks)*Co)
		hidden = self.hidden(x)  # (N, hidden)
		output = self.out(hidden)
		return output
