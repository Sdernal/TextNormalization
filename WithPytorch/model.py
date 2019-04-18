import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import random

START_OF_SENTENCE = 2
END_OF_SENTENCE = 1
PADDING = 0

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, device=None):
        '''
        :param input_size:
        :param hidden_size:
        :param embedding_size:
        '''
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def forward(self, input, hidden):
        # input shape( batch_size, 1)
        batch_size = input.size(0)
        output = self.embedding(input).view( 1,batch_size,-1 )
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=self.device )

    def output_shape(self, batch_size):
        return (1, batch_size, self.hidden_size * 2)


class Decoder(nn.Module):
    """
    Decoder with attention
    """
    def __init__(self, hidden_size, embedding_size, output_size, max_length, dropout_p = 0.1, device=None):
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
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.attn = nn.Linear(hidden_size + embedding_size, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

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
        embedded = self.dropout(embedded)
        # shape (batch_size, max_len)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]),1)), dim=1
        )
        # shapes (batch_size, 1, max_len ) @ (batch_size, max_len, hidden_size * 2) = (batch_size, 1, hidden_size * 2)
        bmm_batch1 = attn_weights.unsqueeze(1)
        bmm_batch2 = encoder_outputs.squeeze(dim=1)
        bmm_batch2 = bmm_batch2.transpose(0, 1)
        # bmm_batch2 = encoder_outputs.view(batch_size, self.max_length, -1)
        attn_applied = torch.bmm(bmm_batch1,bmm_batch2)
        attn_applied = attn_applied.squeeze(dim=1) # (batch_size, hidden_size * 2)
        embedded = embedded.squeeze(dim=0) # (batch_size, embedding_size )
        output = torch.cat((embedded, attn_applied),1) # (batch_size, hidden_size * 2 + embedding_size)
        output = self.attn_combine(output).unsqueeze(0) # (1, batch_size, hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class DecoderCRF(nn.Module):
    """
        Decoder with attention amd CRF
    """

    def __init__(self, hidden_size, embedding_size, output_size, max_length, dropout_p=0.1, device=None):
        '''
        :param hidden_size:
        :param embedding_size:
        :param outpus_size:
        :param max_length:
        '''
        super(DecoderCRF, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.attn = nn.Linear(hidden_size + embedding_size, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        # CRF features
        self.transitions = nn.Parameter(
            torch.randn(self.output_size, self.output_size)
        )
        # TODO: do smth with sos and eos initialisation
        # <SOS> - 2; <EOS> - 1
        # self.transitions.data[START_OF_SENTENCE, :] = -10000
        # self.transitions.data[:, END_OF_SENTENCE] = -10000

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
        embedded = self.dropout(embedded)
        # shape (batch_size, max_len)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        # shapes (batch_size, 1, max_len ) @ (batch_size, max_len, hidden_size * 2) = (batch_size, 1, hidden_size * 2)
        bmm_batch1 = attn_weights.unsqueeze(1)
        bmm_batch2 = encoder_outputs.squeeze(dim=1)
        bmm_batch2 = bmm_batch2.transpose(0, 1)
        # bmm_batch2 = encoder_outputs.view(batch_size, self.max_length, -1)
        attn_applied = torch.bmm(bmm_batch1, bmm_batch2)
        attn_applied = attn_applied.squeeze(dim=1)  # (batch_size, hidden_size * 2)
        embedded = embedded.squeeze(dim=0)  # (batch_size, embedding_size )
        output = torch.cat((embedded, attn_applied), 1)  # (batch_size, hidden_size * 2 + embedding_size)
        output = self.attn_combine(output).unsqueeze(0)  # (1, batch_size, hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

    def output_shape(self, batch_size):
        return (batch_size, self.output_size)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    # CRF stuff:
    def neg_log_likelihood(self, decoder_feats, targets):
        forward_score = self._forward_alg(decoder_feats)
        gold_score = self._score_sentence(decoder_feats, targets)
        return forward_score - gold_score

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.output_size), -10000.)
        # TODO: sos initialistion
        # init_alphas[0][START_OF_SENTENCE] = 0.

        forward_var = torch.Variable(init_alphas)

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.output_size):
                emit_score = feat[next_tag].veiw(1, -1).expand(1, self.output_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var # TODO: + self.transitions[END_OF_SENTENCE]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, targets):
        score = torch.zeros(1)
        for i, feat in enumerate(feats):
            score = score + self.transitions[targets[i+1], targets[i]] + feat[targets[i+1]]
        # TODO: use the terminal symbol in score
        return score

    def _viterbi_decode(self, feats):
        # TODO: realize
        pass


class DecoderPythonCRF(nn.Module):
    '''
    Decoder with python crf unit
    '''
    def __init__(self, hidden_size, embedding_size, output_size, max_length, dropout_p = 0.1,
                 teacher_forcing_ratio = 0.5, device=None):
        '''
        :param hidden_size:
        :param embedding_size:
        :param outpus_size:
        :param max_length:
        '''
        super(DecoderPythonCRF, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.attn = nn.Linear(hidden_size + embedding_size, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.crf = CRF(output_size, batch_first=False)

    def step(self, input, hidden, encoder_outputs):
        batch_size = input.size(1)
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)
        # shape (batch_size, max_len)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        # shapes (batch_size, 1, max_len ) @ (batch_size, max_len, hidden_size * 2) = (batch_size, 1, hidden_size * 2)
        bmm_batch1 = attn_weights.unsqueeze(1)
        bmm_batch2 = encoder_outputs.squeeze(dim=1)
        bmm_batch2 = bmm_batch2.transpose(0, 1)
        # bmm_batch2 = encoder_outputs.view(batch_size, self.max_length, -1)
        attn_applied = torch.bmm(bmm_batch1, bmm_batch2)
        attn_applied = attn_applied.squeeze(dim=1)  # (batch_size, hidden_size * 2)
        embedded = embedded.squeeze(dim=0)  # (batch_size, embedding_size )
        output = torch.cat((embedded, attn_applied), 1)  # (batch_size, hidden_size * 2 + embedding_size)
        output = self.attn_combine(output).unsqueeze(0)  # (1, batch_size, hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

    def forward(self, decoder_input, output_tensor, encoder_outputs, is_test=False):
        output_length = output_tensor.size(0)
        batch_size = decoder_input.size(1)
        decoder_hidden = self.init_hidden(batch_size)
        decoder_outputs = torch.zeros(output_length, batch_size, self.output_size, device=self.device)

        # mask = (output_tensor != 0)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio or is_test else False
        if use_teacher_forcing:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attn = self.step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                true_output = output_tensor[output_item]
                decoder_input = true_output
                decoder_input = decoder_input.view(1, batch_size, 1)
                decoder_outputs[output_item] = decoder_output

        else:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attention = self.step(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_input = decoder_input.view(1, batch_size, 1)
                decoder_outputs[output_item] = decoder_output
        res = self.crf(decoder_outputs, output_tensor, reduction='mean')
        return res

    def predict(self, decoder_input, encoder_outputs, max_output_length):
        output_length = max_output_length
        batch_size = decoder_input.size(1)
        decoder_hidden = self.init_hidden(batch_size)
        decoder_outputs = torch.zeros(output_length, batch_size, self.output_size, device=self.device)
        for output_item in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = self.step(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            decoder_input = decoder_input.view(1, batch_size, 1)
            decoder_outputs[output_item] = decoder_output
        return self.crf.decode(decoder_outputs)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)
