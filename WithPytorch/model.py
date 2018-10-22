import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        '''
        :param input_size:
        :param hidden_size:
        :param embedding_size:
        '''
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        # input shape( batch_size, 1)
        batch_size = input.size(0)
        output = self.embedding(input).view( 1,batch_size,-1 )
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size )


class Decoder(nn.Module):
    """
    Decoder with attention
    """
    def __init__(self, hidden_size, embedding_size, output_size, max_length):
        '''
        :param hidden_size:
        :param embedding_size:
        :param outpus_size:
        :param max_length:
        '''
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.attn = nn.Linear(hidden_size + embedding_size, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param input: torch tensor with shape(1, batch_size, 1)
        :param hidden:
        :param encoder_outputs: torch tensor with shape(max_len, 2, batch_size, hidden_size)
        :return:
        """
        # input shape(1, batch_size, 1)
        # enocoder_outputs_len shape(max_len, hidden_size * 2)
        batch_size = input.size(1)
        embedded = self.embedding(input).view(1, batch_size, -1)
        # dropout
        # shape (batch_size, max_len)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]),1)), dim=1
        )
        # shapes (batch_size, 1, max_len ) @ (batch_size, max_len, hidden_size * 2) = (batch_size, 1, hidden_size * 2)
        bmm_batch1 = attn_weights.unsqueeze(1)
        bmm_batch2 = encoder_outputs.view(batch_size, self.max_length, -1)
        attn_applied = torch.bmm(bmm_batch1,bmm_batch2)
        attn_applied = attn_applied.squeeze() # (batch_size, hidden_size * 2)
        embedded = embedded.squeeze() # (batch_size, embedding_size )
        output = torch.cat((embedded, attn_applied),1) # (batch_size, hidden_size * 2 + embedding_size)
        output = self.attn_combine(output).unsqueeze(0) # (1, batch_size, hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)