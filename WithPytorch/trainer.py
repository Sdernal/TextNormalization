import random
from torch import optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

from entriesprocessor import EntriesProcessor
import numpy as np

class Trainer:
    def __init__(self, encoder, decoder, entries: EntriesProcessor , teacher_forcing_ratio = 0.5, learning_rate=0.01,
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
                true_output = output_tensor[output_item]
                loss += self.criterion(decoder_output, true_output)
                decoder_input = true_output
                decoder_input = decoder_input.view(1, batch_size, 1)
        else:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_input = decoder_input.view(1, batch_size, 1)
                true_output = output_tensor[output_item]
                loss += self.criterion(decoder_output, true_output)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / ( output_length * batch_size )

    def test_training(self):
        BATCH_SIZE = 10
        input_tensor = torch.ones(self.max_input_length, BATCH_SIZE, 1, dtype=torch.long)
        output_tensor = torch.ones(self.max_output_length, BATCH_SIZE, dtype=torch.long)
        self.train_on_batch(input_tensor, output_tensor)

    def iteate_batches(self, batch_size=100):
        n_samples = self.entries.X_data.shape[0]
        x_len = self.entries.X_data.shape[1]
        y_len = self.entries.y_data.shape[1]

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = self.entries.X_data[batch_idx, ]
            Y_batch = self.entries.y_data[batch_idx, ]

            X_batch = torch.from_numpy(X_batch)
            Y_batch = torch.from_numpy(Y_batch)

            X_batch =  X_batch.type(torch.LongTensor)
            Y_batch = Y_batch.type(torch.LongTensor)

            X_batch = X_batch.view(x_len, -1, 1)
            Y_batch = Y_batch.view(y_len, -1)
            yield Variable(X_batch), Variable(Y_batch)

    def train_epoch(self):
        '''
        Train on dataset for one epoch
        :return: None
        '''
        train_count, train_loss = 0, 0
        start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(self.iteate_batches()):
            loss = self.train_on_batch(x_batch, y_batch)
            train_loss += loss
            train_count = i + 1
            current_time = time.time()
            print('\r\tTrain Loss: %6f \t Time: %ds' % (train_loss / train_count, current_time - start_time), end='')
        print()

    def train(self):
        print()
        for epoch in range(1,6):
            print('Epoch %d' % (epoch))
            self.train_epoch()
