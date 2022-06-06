import logging

import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor


def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if isinstance(h, Variable):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


class lstm(nn.Module):
  def __init__(self, vocab_size, embedding_dim=1500, num_steps=35, batch_size=20, num_layers=2, dp_keep_prob=0.35):
    super(lstm, self).__init__()
    logging.info("vocab_size: {}, batch_size :{}, num_steps:{} ".format(
      vocab_size, batch_size, num_steps))
    self.embedding_dim = embedding_dim
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=num_layers,
                            dropout=1 - dp_keep_prob)
    self.sm_fc = nn.Linear(in_features=embedding_dim,
                           out_features=vocab_size)
    self.sm_fc.weight = self.word_embeddings.weight
    self.init_weights()

  def init_weights(self):
    init_range = 0.1
    self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()),
            Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()))

  def forward(self, inputs, hidden):
    # inputs = inputs.reshape(self.num_steps, self.batch_size)
    inputs = inputs.transpose(0, 1)
    # logging.info("inputs: shape {}".format(inputs.shape))
    embeds = self.dropout(self.word_embeddings(inputs))
    # logging.info("embeds: shape {}".format(embeds.shape))
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = self.dropout(lstm_out)
    logits = self.sm_fc(lstm_out.view(-1, self.embedding_dim))
    return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden


class lstmwt2(nn.Module):
  def __init__(self, vocab_size, embedding_dim=1500, num_steps=35, batch_size=20, num_layers=3, dp_keep_prob=0.5):
    super(lstmwt2, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_size = 650
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=num_layers,
                            dropout=1 - dp_keep_prob)
    self.sm_fc = nn.Linear(in_features=self.hidden_size,
                           out_features=vocab_size)
    self.init_weights()

  def init_weights(self):
    init_range = 0.1
    self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()),
            Variable(weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()))

  def forward(self, inputs, hidden):
    embeds = self.dropout(self.word_embeddings(inputs))
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = self.dropout(lstm_out)
    logits = self.sm_fc(lstm_out.view(-1, self.hidden_size))
    return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden



