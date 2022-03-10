import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class CharLSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim, dropout,
                 vocab_size, alphabet_size, num_kernels, kernel_width, directions=1, device='cpu',
                 pos_vector_size=0):

        # unused args are there only to share an API with the other base tagger classes

        super(CharLSTMTagger, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.pos_vector_size = pos_vector_size
        self.directions = directions
        self.dropout = nn.Dropout(dropout)

        self.char_embeddings = nn.Embedding(alphabet_size, char_embedding_dim)
        # we don't apply dropout in the char lstm for now
        self.lstm_char = nn.LSTM(char_embedding_dim, char_embedding_dim, bidirectional=directions == 2)

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm_word = nn.LSTM(word_embedding_dim + self.pos_vector_size + char_embedding_dim * directions,
                                 hidden_dim, dropout=dropout, bidirectional=directions == 2)

        self.hidden = self.init_hidden(hidden_dim)
        self.hidden_char = self.init_hidden(char_embedding_dim)

    def init_hidden(self, dim):
        return (torch.zeros(self.directions, 1, dim).to(device=self.device),
                torch.zeros(self.directions, 1, dim).to(device=self.device))

    def forward(self, sentence):
        word_idxs = []
        lstm_char_result = []
        pos_idxs = []
        for word in sentence:
            self.hidden_char = self.init_hidden(self.char_embedding_dim)
            word_idxs.append(word[0])
            if self.pos_vector_size:
                pos_idxs.append(word[2])
            char_idx = torch.LongTensor(word[1]).to(device=self.device)
            char_embeds = self.char_embeddings(char_idx)
            lstm_char_out, self.hidden_char = self.lstm_char(char_embeds.view(len(word[1]), 1, self.char_embedding_dim),
                                                             self.hidden_char)
            lstm_char_result.append(lstm_char_out[-1])

        lstm_char_result = torch.stack(lstm_char_result)

        word_embeds = self.word_embeddings(torch.LongTensor(word_idxs).to(device=self.device)).view(
            len(sentence), 1, self.word_embedding_dim)

        if self.pos_vector_size:  # if the pos vector size is not zero, we concatenate POS information to embeddings
            one_hot_buffer = torch.eye(self.pos_vector_size).to(device=self.device)
            indices = torch.LongTensor(pos_idxs).to(device=self.device)
            one_hots = one_hot_buffer[indices].view(len(sentence), 1, self.pos_vector_size)
            word_embeds = torch.cat((word_embeds, one_hots), 2)

        lstm_in = torch.cat((word_embeds, lstm_char_result), 2)

        lstm_out, self.hidden = self.lstm_word(lstm_in, self.hidden)
        lstm_out = self.dropout(lstm_out)

        return lstm_out
