import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        # input shape( batch_size, 1)
        batch_size = input.size(0)
        output = self.embedding(input).view( 1,batch_size,-1 )
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size )


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, outpus_size, max_length):
        self.hidden_size = hidden_size
        self.output_size = outpus_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.attn = nn.Linear(hidden_size + embedding_size, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, outpus_size)

    def forward(self, input, hidden, encoder_outputs):
        # input shape(batch_size, 1)
        # enocoder_outputs_len shape(max_len, hidden_size * 2)
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, -1)
        # dropout
        # shape (batch_size, max_len)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]),1)), dim=1
        )
        # shapes (1, batch_size, max_len ) @ (1, max_len, hidden_size * 2) = (1, batch_size, hidden_size * 2)
        attn_applied = torch.bmm(attn_weights.usqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]),1)
        output = self.attn_combine(output).usqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)