import random
from torch import optim
from entriesprocessor import EntriesProcessor
import torch

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
