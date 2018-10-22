import random
from torch import optim
import torch
import torch.nn as nn
class Trainer:
    def __init__(self, encoder, decoder, entries, teacher_forcing_ratio = 0.5, learning_rate=0.01,
                 max_input_length = 40, max_output_length = 20):
        self.encoder = encoder
        self.decoder = decoder
        self.entries = entries
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def train_on_batch(self, input_tensor, output_tensor):
        '''
        :param input_tensor: torch tensor with size(max_length, batch_size, 1)
        :param output_tensor: torch tensor with size(max_length, batch_size, 1)
        :return: None
        '''
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        output_length = output_tensor.size(0)
        assert input_tensor.size(1) == output_tensor.size(1)
        batch_size = input_tensor.size(1)
        loss = 0
        encoder_hidden = self.encoder.init_hidden(batch_size)
        # encoder_outputs.size() = (max_len, 2, batch_size, hidden_size)
        encoder_outputs = torch.zeros(self.max_input_length, *encoder_hidden.size())
        for input_item in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[input_item], encoder_hidden
            )
            encoder_outputs[input_item] = encoder_hidden

        decoder_input = torch.ones(1, batch_size, 1, dtype=torch.long) * self.entries.symbols_dict['<PAD>']
        decoder_hidden = self.decoder.init_hidden(batch_size)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                loss += self.criterion(decoder_output, output_tensor[output_item])
                decoder_input = output_tensor[output_item]
        else:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += self.criterion(decoder_output, output_tensor[output_item])
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / ( output_length * batch_size )

    def test_training(self):
        BATCH_SIZE = 10
        input_tensor = torch.ones(self.max_input_length, BATCH_SIZE, 1, dtype=torch.long)
        output_tensor = torch.ones(self.max_output_length, BATCH_SIZE, 1, dtype=torch.long)
        self.train_on_batch(input_tensor, output_tensor)