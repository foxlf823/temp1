''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


class Transformer_Pooling(nn.Module):

    def __init__(
            self,
            n_src_vocab, num_classes, len_max_seq, gpu,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()

        self.gpu = gpu

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab.vocab_size, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder.src_word_emb = n_src_vocab.init_embed_layer()
        # if self.gpu >= 0 and torch.cuda.is_available():
        #     self.encoder.src_word_emb = self.encoder.src_word_emb.cuda(self.gpu)

        self.tgt_word_prj = nn.Linear(d_model, num_classes, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.encoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.



    def forward(self, input, char_inputs):

        entity_words, word_position, entity_lengths, entity_seq_recover = input

        enc_output, *_ = self.encoder(entity_words, word_position)

        # _, max_len, dim = enc_output.size()
        # idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        # if self.gpu >= 0 and torch.cuda.is_available():
        #     idxes = idxes.cuda(self.gpu)
        # mask = (idxes < entity_lengths.unsqueeze(1)).float().unsqueeze(-1).expand(-1, -1, dim)
        #
        # enc_output = enc_output*mask

        batch_size, max_len, _ = enc_output.size()
        pooled_enc_output = nn.functional.avg_pool2d(enc_output.unsqueeze(1), (max_len, 1)).view(batch_size, -1)


        seq_logit = self.tgt_word_prj(pooled_enc_output) * self.x_logit_scale

        return seq_logit


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size, gpu):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        self.gpu = gpu

    def forward(self, inputs, lengths):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = nn.functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if self.gpu >= 0 and torch.cuda.is_available():
            idxes = idxes.cuda(self.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output


class Transformer_Attn(nn.Module):

    def __init__(
            self,
            n_src_vocab, num_classes, len_max_seq, gpu,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, d_hidden=100,
            ):

        super().__init__()

        self.gpu = gpu

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab.vocab_size, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=0.1)

        self.encoder.src_word_emb = n_src_vocab.init_embed_layer()

        self.attn = DotAttentionLayer(d_model, gpu)

        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(d_model, d_hidden)

        self.tgt_word_prj = nn.Linear(d_hidden, num_classes)


    def forward(self, input, char_inputs):

        entity_words, word_position, entity_lengths, entity_seq_recover = input

        enc_output, *_ = self.encoder(entity_words, word_position)

        batch_size, max_len, _ = enc_output.size()

        pooled_enc_output = self.attn(enc_output, entity_lengths)

        output = self.dropout(pooled_enc_output)
        output = self.hidden(output)

        seq_logit = self.tgt_word_prj(output)

        return seq_logit

class CharCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(CharCNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

    def get_last_hiddens(self, input):

        batch_size = input.size(0)
        char_embeds = input.transpose(2,1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = nn.functional.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

class Transformer_Attn_CNN(nn.Module):

    def __init__(
            self,
            n_src_vocab, num_classes, len_max_seq, gpu, char_vocab, len_max_char,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, d_hidden=100
            ):

        super().__init__()

        self.gpu = gpu

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab.vocab_size, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=0.1)

        self.encoder.src_word_emb = n_src_vocab.init_embed_layer()

        self.char_encoder = Encoder(
            n_src_vocab=char_vocab.vocab_size, len_max_seq=len_max_char,
            d_word_vec=char_vocab.emb_size, d_model=char_vocab.emb_size, d_inner=4*char_vocab.emb_size,
            n_layers=n_layers, n_head=8, d_k=char_vocab.emb_size//8, d_v=char_vocab.emb_size//8,
            dropout=0.1)

        self.char_encoder.src_word_emb = char_vocab.init_embed_layer()

        self.char_feature = CharCNN(char_vocab.emb_size, char_vocab.emb_size)

        self.input_size = d_model + char_vocab.emb_size

        self.attn = DotAttentionLayer(self.input_size, gpu)

        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(self.input_size, d_hidden)

        self.tgt_word_prj = nn.Linear(d_hidden, num_classes)


    def forward(self, input, char_inputs):
        entity_words, word_position, entity_lengths, entity_seq_recover = input

        batch_size, max_len = entity_words.size()

        enc_output, *_ = self.encoder(entity_words, word_position)

        char_inputs, char_position, char_seq_lengths, char_seq_recover = char_inputs
        char_enc_output, *_ = self.char_encoder(char_inputs, char_position)
        char_features = self.char_feature.get_last_hiddens(char_enc_output)
        char_features = char_features[char_seq_recover]
        char_features = char_features.view(batch_size, max_len, -1)

        inputs = torch.cat((enc_output, char_features), 2)

        pooled_enc_output = self.attn(inputs, entity_lengths)

        output = self.dropout(pooled_enc_output)
        output = self.hidden(output)

        seq_logit = self.tgt_word_prj(output)

        return seq_logit