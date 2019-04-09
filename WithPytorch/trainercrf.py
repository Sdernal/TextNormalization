import random
from torch import optim
from entriesprocessor import EntriesProcessor
import torch
from dataloader import distance
from torch.autograd import Variable

class TrainerCRF:
    def __init__(self, encoder, decoder, entries: EntriesProcessor,teacher_forcing_ratio = 0.5, learning_rate=0.01,
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
        loss = 0
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs = torch.zeros(self.max_input_length, *self.encoder.output_shape(batch_size), device=self.device)

        for input_item in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[input_item], encoder_hidden
            )
            encoder_outputs[input_item] = encoder_output

        decoder_input = torch.ones(1, batch_size, 1, dtype=torch.long, device=self.device) * self.entries.symbols_dict['<PAD>']
        decoder_hidden = self.decoder.init_hidden(batch_size)
        # save decoder output rnn features to process them through CRF

        decoder_outputs = torch.zeros(output_length, *self.decoder.output_shape(batch_size))
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # save the decoder feats
                decoder_outputs[output_length] = decoder_output
                true_output = output_tensor[output_item]
                decoder_input = true_output
                decoder_input = decoder_input.view(1, batch_size, 1)
        else:
            for output_item in range(output_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # save the decoder feats
                decoder_outputs[output_length] = decoder_output
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_input = decoder_input.view(1, batch_size, 1)
                true_output = output_tensor[output_item]

        # use CRF loglikelihood loss
        # TODO: debug this
        loss = self.decoder.neg_log_likelihood(decoder_outputs, output_tensor)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / ( output_length )


    def test_model(self, from_train=True):
        if from_train:
            X_data = self.entries.X_data_train
            Y_data = self.entries.y_data_train
        else:
            X_data = self.entries.X_data_test
            Y_data = self.entries.y_data_test

        n_samples = X_data.shape[0]
        ethalons = []
        results = []
        inputs = []
        matched = []
        distances = []

        for i in range(n_samples):
            input, result = self.evaluate_sample(X_data[i:i + 1])

            ethalon = Y_data[i:i + 1]
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
            decoder_hidden = self.decoder.init_hidden(batch_size)

            result = ''

            self.predict
            for output_item in range(self.max_output_length):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_input = decoder_input.view(1, batch_size, 1)
                result_symbol = self.entries.symbols_dict_rev[decoder_input.view(1).item()]
                if len(result_symbol) == 1:
                    result += result_symbol

            return input, result