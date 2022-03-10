import torch
import torch.nn as nn

import math


class CharCNNTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim, dropout,
                 vocab_size, alphabet_size, num_kernels, kernel_width,
                 directions=1, device='cpu', pos_vector_size=0):
        super(CharCNNTagger, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.pos_vector_size = pos_vector_size
        self.directions = directions
        self.dropout = nn.Dropout(dropout)

        self.char_embeddings = nn.Embedding(alphabet_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm_word = nn.LSTM(word_embedding_dim + num_kernels + self.pos_vector_size, hidden_dim, dropout=dropout,
                                 bidirectional=directions == 2)

        self.hidden = self.init_hidden(hidden_dim)
        self.hidden_char = self.init_hidden(char_embedding_dim)

        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        padding_size = math.ceil((kernel_width-1)/2)
        self.conv = nn.Conv1d(self.char_embedding_dim,
                        self.num_kernels, self.kernel_width,
                        padding=padding_size)

    def init_hidden(self, dim):
        return (torch.zeros(self.directions, 1, dim).to(device=self.device),
                torch.zeros(self.directions, 1, dim).to(device=self.device))

    def forward(self, sentence):
        word_idxs = []
        pos_idxs = []
        char_cnn_result = []
        for word in sentence:
            self.hidden_char = self.init_hidden(self.char_embedding_dim)
            word_idxs.append(word[0])
            if self.pos_vector_size:
                pos_idxs.append(word[2])  # word[2] contains the POS tag of the word
            char_idx = torch.LongTensor(word[1]).to(device=self.device)
            char_embeds = self.char_embeddings(char_idx)
            char_embeds = char_embeds.transpose(0, 1).contiguous()
            char_embeds.unsqueeze_(0)
            conv_out = self.conv(char_embeds)
            pool_out = torch.max(torch.tanh(conv_out), 2)[0]
            char_cnn_result.append(pool_out)

        char_cnn_result = torch.stack(char_cnn_result)

        word_embeds = self.word_embeddings(torch.LongTensor(word_idxs).to(
            device=self.device)).view(len(sentence), 1, self.word_embedding_dim)

        if self.pos_vector_size:  # if the pos vector size is not zero, we concatenate POS information to embeddings
            one_hot_buffer = torch.eye(self.pos_vector_size).to(device=self.device)
            indices = torch.LongTensor(pos_idxs).to(device=self.device)
            one_hots = one_hot_buffer[indices].view(len(sentence), 1, self.pos_vector_size)
            word_embeds = torch.cat((word_embeds, one_hots), 2)

        lstm_in = torch.cat((word_embeds, char_cnn_result), 2)

        lstm_out, self.hidden = self.lstm_word(lstm_in, self.hidden)
        lstm_out = self.dropout(lstm_out)

        return lstm_out
