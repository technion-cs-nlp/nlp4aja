import torch
import torch.nn as nn


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, dropout, vocab_size, alphabet_size,
                 num_kernels, kernel_width, directions=1, device='cpu', pos_vector_size=0):

        # unused args are there only to share an API with the other base tagger classes
        super(LSTMTagger, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.directions = directions
        self.dropout = nn.Dropout(dropout)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_vector_size = pos_vector_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim + self.pos_vector_size, hidden_dim, dropout=dropout,
                            bidirectional=directions == 2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden = self.init_hidden(hidden_dim)

    def init_hidden(self, dim):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the PyTorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.directions, 1, dim).to(device=self.device),
                torch.zeros(self.directions, 1, dim).to(device=self.device))

    def forward(self, sentence):
        pos_idxs = []
        word_idxs = []
        for word in sentence:
            word_idxs.append(word[0])
            if self.pos_vector_size:
                pos_idxs.append(word[1])

        embeds = self.word_embeddings(torch.LongTensor(word_idxs).to(device=self.device))
        if self.pos_vector_size:  # if the pos vector size is not zero, we concatenate POS information to embeddings
            one_hot_buffer = torch.eye(self.pos_vector_size)
            indices = torch.LongTensor(pos_idxs)
            one_hots = one_hot_buffer[indices].view(len(sentence), 1, self.pos_vector_size)
            embeds = torch.cat((embeds, one_hots), 2)

        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = self.dropout(lstm_out)
        return lstm_out
