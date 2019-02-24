import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt
from model.charcnn import CharCNN

class CNNCNN_SentCNN(nn.Module):
	def __init__(self, vocab, num_classes, char_alphabet):

		super(CNNCNN_SentCNN, self).__init__()
		self.embedding = vocab.init_embed_layer()
		self.hidden_size = opt.hidden_size

		# charcnn
		self.char_hidden_dim = 10
		self.char_embedding_dim = 20
		self.char_feature = CharCNN(len(char_alphabet), None, self.char_embedding_dim, self.char_hidden_dim,
									opt.dropout, opt.gpu)

		self.embedding_size = self.embedding.weight.size(1)
		self.hidden_size = opt.hidden_size

		Ci = 1
		Co = opt.kernel_num
		Ks = [int(k) for k in list(opt.kernel_sizes) if k != ","]
		#mention char_cnn
		D = self.embedding_size + self.char_hidden_dim
		self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
											   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
		self.mention_hidden = nn.Linear(len(Ks) * Co, self.hidden_size)

		# sentence cnn
		self.sent_hidden_size = opt.sent_hidden_size
		self.sent_convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, self.embedding_size), stride=(1, 1),
											   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
		self.sent_hidden = nn.Linear(len(Ks) * Co, self.sent_hidden_size)


		self.hidden = nn.Linear(self.hidden_size+self.sent_hidden_size, self.hidden_size)#mention_hidden_size + sentence_hidden_size
		self.out = nn.Linear(self.hidden_size, num_classes)
		self.dropout = nn.Dropout(opt.dropout)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, mention_inputs, char_inputs, sent_inputs):
		inputs, lengths, seq_recover = mention_inputs
		mention_embedding = self.embedding(inputs)# (N, W, D)

		batch_size, max_len = inputs.size()
		char_inputs, char_seq_lengths, char_seq_recover = char_inputs
		char_features = self.char_feature.get_last_hiddens(char_inputs)
		char_features = char_features[char_seq_recover]
		char_features = char_features.view(batch_size, max_len, -1)
		mention_char = torch.cat((mention_embedding, char_features), 2)
		mention_char = mention_char.unsqueeze(1)  # (N, Ci, W, D)
		mention_char = [F.relu(conv(mention_char)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
		mention_char = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in mention_char]  # [(N, Co), ...]*len(Ks)
		mention_char = torch.cat(mention_char, 1)
		mention_hidden = self.mention_hidden(mention_char)

		sent_inputs, sent_seq_lengths = sent_inputs
		sent_embedding =self.embedding(sent_inputs)
		sent_feature = sent_embedding.unsqueeze(1)  # (N, Ci, W, D)
		sent_feature = [F.relu(conv(sent_feature)).squeeze(3) for conv in self.sent_convs1]  # [(N, Co, W), ...]*len(Ks)
		sent_feature = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in sent_feature]  # [(N, Co), ...]*len(Ks)
		sent_feature = torch.cat(sent_feature, 1)
		sent_hidden = self.sent_hidden(sent_feature)

		x = torch.cat((mention_hidden, sent_hidden), 1)
		x = self.dropout(x)  # (N, len(Ks)*Co)
		hidden = self.hidden(x)  # (N, hidden)
		output = self.out(hidden)
		return output
