import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from char_cnn_tagger import CharCNNTagger

WORD_LSTM = "word_lstm"
CHAR_LSTM = "char_lstm"
CHAR_CNN = "char_cnn"


class MTLWrapper(nn.Module):
    
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim, dropout,
                 vocab_size, alphabet_size, tagset_sizes, num_kernels, kernel_width, directions=1, device='cpu',
                 pos_dict_size=0, model_type=CharCNNTagger):
        """
        This is a wrapper for MTL learning, that receives a base model type, and a list of tagset sizes, to train
        the name number of models on each of the tagsets. When len(tagset_sizes) == 1, this will be the same as though
        the base model was not wrapped by anything.
        :param word_embedding_dim:
        :param char_embedding_dim:
        :param hidden_dim:
        :param dropout:
        :param vocab_size:
        :param alphabet_size:
        :param tagset_sizes:
        :param num_kernels:
        :param kernel_width:
        :param directions:
        :param device:
        :param pos_dict_size:
        :param model_type:
        """
        super(MTLWrapper, self).__init__()
        self.base_model = model_type(word_embedding_dim, char_embedding_dim, hidden_dim, dropout, vocab_size,
                                     alphabet_size, num_kernels, kernel_width, directions, device,
                                     pos_vector_size=pos_dict_size)
        self.linear_taggers = []
        for tagset_size in tagset_sizes:
            self.linear_taggers.append(nn.Linear(hidden_dim*directions, tagset_size).to(device=device))
        self.linear_taggers = nn.ModuleList(self.linear_taggers)

    def init_hidden(self, hidden_dim):
        """
        For multitask models, the hidden dimension is shared across all the tasks
        :param hidden_dim:
        :return:
        """
        self.base_model.hidden = self.base_model.init_hidden(hidden_dim)
        return self.base_model.init_hidden(hidden_dim)

    def forward(self, sentence):
        """
        Calls forward on each of the models in turn and returns list of the scores
        :param sentence:
        :return:
        """
        lstm_out = self.base_model.forward(sentence)
        tag_scores = []
        for linear_tagger in self.linear_taggers:
            tag_space = linear_tagger(lstm_out.view(len(sentence), -1))
            tag_scores.append(F.log_softmax(tag_space))
        # tag_scores = torch.stack(tag_scores)
        return tag_scores
