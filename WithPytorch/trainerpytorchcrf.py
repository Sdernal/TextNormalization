import random
from torch import optim
from entriesprocessor import EntriesProcessor
import torch
from model import DecoderPythonCRF, Encoder
from torch.autograd import Variable
import numpy as np
import time
from dataloader import distance


class TrainerPythonCRF:
    def __init__(self, encoder: Encoder, decoder: DecoderPythonCRF, entries: EntriesProcessor,teacher_forcing_ratio = 0.5, learning_rate=0.01,
                 max_input_length=40, max_output_length=20, device=None):
        self.encoder = encoder
        self.decoder = decoder
        self.entries = entries
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
        self.decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        if device is None:
            self.device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
        else:
            self.device = device

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
        # loss = 0
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs = torch.zeros(self.max_input_length, *self.encoder.output_shape(batch_size), device=self.device)

        for input_item in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[input_item], encoder_hidden
            )
            encoder_outputs[input_item] = encoder_output
        decoder_input = torch.ones(1, batch_size, 1, dtype=torch.long, device=self.device) * self.entries.symbols_dict[
            '<PAD>']
        loss = -self.decoder(decoder_input, output_tensor, encoder_outputs)

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def train(self, num_epoches, batch_size=100):
        print()
        train_losses = []
        test_losses = []
        for epoch in range(1,num_epoches + 1):
            print('Epoch %d' % (epoch))
            train_losses.append(self.train_epoch(batch_size))
            # test_losses.append(self.test_epoch(batch_size))

        return train_losses, test_losses

    def iteate_batches(self, batch_size=100, is_train=True):
        if is_train:
            X_data = self.entries.X_data_train
            Y_data = self.entries.y_data_train
        else:
            X_data = self.entries.X_data_test
            Y_data = self.entries.y_data_test

        n_samples = X_data.shape[0]
        x_len = X_data.shape[1]
        y_len = Y_data.shape[1]

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = X_data[batch_idx, ].T
            Y_batch = Y_data[batch_idx, ].T

            X_batch = torch.from_numpy(X_batch)
            Y_batch = torch.from_numpy(Y_batch)

            X_batch =  X_batch.type(torch.LongTensor)
            Y_batch = Y_batch.type(torch.LongTensor)

            X_batch = X_batch.view(x_len, -1, 1)
            Y_batch = Y_batch.view(y_len, -1)

            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            yield Variable(X_batch), Variable(Y_batch)

    def train_epoch(self,  batch_size=100):
        '''
        Train on dataset for one epoch
        :return: None
        '''
        train_count, train_loss = 0, 0
        start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(self.iteate_batches(batch_size=batch_size)):
            loss = self.train_on_batch(x_batch, y_batch)
            train_loss += loss
            train_count = i + 1
            current_time = time.time()
            print('\r\tTrain Loss: %6f \t Time: %ds' % (train_loss / train_count, current_time - start_time), end='')
        print()
        return train_loss / train_count

    def test_model(self, from_train=True, n_samples = None):
        if from_train:
            X_data = self.entries.X_data_train
            Y_data = self.entries.y_data_train
        else:
            X_data = self.entries.X_data_test
            Y_data = self.entries.y_data_test

        n_samples = X_data.shape[0] if n_samples is None else min(n_samples, X_data.shape[0])
        ethalons = []
        results = []
        inputs = []
        matched = []
        distances = []

        for i in range(n_samples):
            input, result = self.evaluate_sample(X_data[i:i + 1])

            result = result[0]
            ethalon = Y_data[i:i + 1]
            result = [self.entries.symbols_dict_rev[result[i]] for i in range(len(result))]
            ethalon = [self.entries.symbols_dict_rev[ethalon[0][i]] for i in range(ethalon.shape[1])]
            ethalon = filter(lambda x: len(x) == 1, ethalon)
            ethalon = ''.join(ethalon)

            ethalons.append(ethalon)
            results.append(result)
            inputs.append(input)
            matched.append(ethalon == result)
            distances.append(distance(result, ethalon))

        return ethalons, results, inputs, matched, distances

    def evaluate_sample(self, data):
        with torch.no_grad():
            # data = self.entries.X_data[item:item+1]
            input = [self.entries.symbols_dict_rev[data[0][i]] for i in range(data.shape[1])]
            input = filter(lambda x: len(x) == 1, input)
            input = ''.join(input)
            data = torch.from_numpy(data)
            data = data.type(torch.LongTensor)
            data = data.to(self.device)
            data = data.view(-1, 1, 1)
            input_tensor = Variable(data)
            input_length = input_tensor.size(0)
            batch_size = input_tensor.size(1)
            loss = 0
            encoder_hidden = self.encoder.init_hidden(batch_size)
            # encoder_outputs.size() = (max_len, 2, batch_size, hidden_size)
            encoder_outputs = torch.zeros(self.max_input_length, *self.encoder.output_shape(batch_size),
                                          device=self.device)
            for input_item in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[input_item], encoder_hidden
                )
                encoder_outputs[input_item] = encoder_output

            decoder_input = torch.ones(1, batch_size, 1, dtype=torch.long, device=self.device) * \
                            self.entries.symbols_dict['<PAD>']

            return input, self.decoder.predict(decoder_input, encoder_outputs, self.max_output_length)

