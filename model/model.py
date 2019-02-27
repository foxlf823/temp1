import torch
import torch.nn as nn
from options import opt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as functional
from model.charbilstm import CharBiLSTM



class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, lengths):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if opt.gpu >= 0 and torch.cuda.is_available():
            idxes = idxes.cuda(opt.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output


class BiLSTM_Attn(nn.Module):
    def __init__(self, vocab, num_classes, char_alphabet):
        super(BiLSTM_Attn, self).__init__()

        self.embed_size = opt.word_emb_size
        self.embedding = vocab.init_embed_layer()
        self.hidden_size = opt.hidden_size
        self.char_hidden_dim = 10
        self.char_embedding_dim = 20
        self.char_feature = CharBiLSTM(len(char_alphabet), None, self.char_embedding_dim, self.char_hidden_dim,
									opt.dropout, opt.gpu)

        self.input_size = self.embed_size + self.char_hidden_dim
        self.lstm_hidden = self.hidden_size // 2

        self.lstm = nn.LSTM(self.input_size, self.lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)

        self.attn = DotAttentionLayer(self.hidden_size)

        self.hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input, char_inputs):
        entity_words, entity_lengths, entity_seq_recover = input

        entity_words_embeds = self.embedding(entity_words)
        batch_size, max_len, _ = entity_words_embeds.size()

        char_inputs, char_seq_lengths, char_seq_recover = char_inputs
        char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths)
        char_features = char_features[char_seq_recover]
        char_features = char_features.view(batch_size, max_len, -1)

        input_embeds = torch.cat((entity_words_embeds, char_features), 2)

        packed_words = pack_padded_sequence(input_embeds, entity_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        output = self.attn(lstm_out.transpose(1, 0), entity_lengths)

        output = self.dropout(output)

        output = self.hidden(output)
        output = self.out(output)
        return output