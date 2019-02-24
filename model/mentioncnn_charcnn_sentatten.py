import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from options import opt
from model.charcnn import CharCNN

class CNNCNN_SentATTEN(nn.Module):
	def __init__(self, vocab, num_classes, char_alphabet):
		super(CNNCNN_SentATTEN,self).__init__()
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
		# mention char_cnn
		D = self.embedding_size + self.char_hidden_dim
		self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
											   padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
		self.mention_hidden = nn.Linear(len(Ks) * Co, self.hidden_size)

		#sentence atten
		self.atten_W = nn.Linear(self.embedding_size, 1, bias=False)
		self.sent_hidden_size = opt.sent_hidden_size
		self.sent_hidden = nn.Linear(self.embedding_size, self.sent_hidden_size)
		self.hidden = nn.Linear(self.hidden_size + self.sent_hidden_size, self.hidden_size)  # mention_hidden_size + sentence_hidden_size
		self.out = nn.Linear(self.hidden_size, num_classes)
		self.dropout = nn.Dropout(opt.dropout)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, mention_inputs, char_inputs, sent_inputs):
		inputs, lengths, seq_recover = mention_inputs
		mention_embedding = self.embedding(inputs)  # (N, W, D)

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
		sent_embedding = self.embedding(sent_inputs)
		sent_batch_size, sent_max_len, _ = sent_embedding.size()
		flat_input = sent_embedding.contiguous().view(-1, self.embedding_size)
		logits = self.atten_W(flat_input).view(sent_batch_size, sent_max_len)
		alphas = F.softmax(logits, dim=1)

		# computing mask
		idxes = torch.arange(0, sent_max_len, out=torch.LongTensor(sent_max_len)).unsqueeze(0).cuda(opt.gpu)
		mask = autograd.Variable((idxes < sent_seq_lengths.unsqueeze(1)).float())

		alphas = alphas * mask
		# renormalize
		alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
		sent_atten_input = torch.bmm(alphas.unsqueeze(1), sent_embedding).squeeze(1)
		sent_atten_input = self.dropout(sent_atten_input)
		sent_hidden = self.sent_hidden(sent_atten_input)

		x = torch.cat((mention_hidden, sent_hidden), 1)
		x = self.dropout(x)
		hidden = self.hidden(x)  # (N, hidden)
		output = self.out(hidden)
		return output
