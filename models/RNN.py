import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class RNN(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(RNN, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.rnn = nn.RNN(embedding_length, hidden_size, num_layers=2, bidirectional=True)
		self.label = nn.Linear(4*hidden_size, output_size)
	
	def forward(self, input_sentences, batch_size=None):
		
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the final_hidden_state of RNN.
		logits.size() = (batch_size, output_size)
		
		"""

		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # 4 = num_layers*num_directions
		else:
			h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())
		output, h_n = self.rnn(input, h_0)
		# h_n.size() = (4, batch_size, hidden_size)
		h_n = h_n.permute(1, 0, 2) # h_n.size() = (batch_size, 4, hidden_size)
		h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
		# h_n.size() = (batch_size, 4*hidden_size)
		logits = self.label(h_n) # logits.size() = (batch_size, output_size)
		
		return logits
