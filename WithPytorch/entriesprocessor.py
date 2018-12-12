# import dataloader
import numpy as np


class EntriesProcessor:

    def __init__(self, max_value_len=20, max_context_len=40, window_size=10):
        """
        :param max_value_len: max length of normalized entity
        :param max_context_len: max length of context window
        """
        self.uniq_symbols = set()
        self.symbols_dict = {'<PAD>': 0, '<EOS>': 1, '<SOS>': 2, '<UNK>': 3}
        self.symbols_dict_rev = {0: '<PAD>', 1: '<EOS>', 2: '<SOS>', 3: '<UNK>'}
        self.symbols_counter = 3
        self.X_data_train = None
        self.y_data_train = None
        self.X_data_test = None
        self.y_data_test = None
        self.MAX_VALUE_LEN = max_value_len
        self.MAX_CONTEXT_LEN = max_context_len
        self.WINDOW_SIZE = window_size

    def process(self, entries, train_ratio=0.9):
        for entry in entries:
            self.uniq_symbols |= set(entry.value.lower())
            self.uniq_symbols |= set(entry.context.lower())

        for symbol in self.uniq_symbols:
            if symbol not in self.symbols_dict:
                self.symbols_dict[symbol] = self.symbols_counter
                self.symbols_dict_rev[self.symbols_counter] = symbol
                self.symbols_counter += 1

        #         context_lengths = list(map(lambda x: len(x.context), entries))
        #         value_lengths = list(map(lambda x: len(x.value), entries))
        uniq_values = list(set([entry.value for entry in entries]))

        train_count = int(len(uniq_values) * train_ratio)
        uniq_train_values = set(uniq_values[:train_count])
        uniq_test_values = set(uniq_values[train_count:])

        train_contexts, train_values = [], []
        test_contexts, test_values = [], []
        for entry in entries:
            left_paddings_count = self.WINDOW_SIZE - (entry.offset - entry.context_offset)
            context = ([self.symbols_dict['<PAD>']] * (left_paddings_count + 1) + [self.symbols_dict['<SOS>']] +
                       list(map(lambda x: self.symbols_dict[x], entry.context.lower())) + [self.symbols_dict['<EOS>']])
            value = (list(map(lambda x: self.symbols_dict[x], entry.value.lower())) + [self.symbols_dict['<EOS>']])
            if len(context) > self.MAX_CONTEXT_LEN or len(value) > self.MAX_VALUE_LEN:
                continue

            if entry.value in uniq_train_values:
                train_contexts.append(context)
                train_values.append(value)
            else:
                test_contexts.append(context)
                test_values.append(value)

        assert len(train_contexts) == len(train_values)
        assert len(test_contexts) == len(test_values)

        self.X_data_train = np.zeros((len(train_contexts), self.MAX_CONTEXT_LEN))
        self.y_data_train = np.zeros((len(train_values), self.MAX_VALUE_LEN))

        self.X_data_test = np.zeros((len(test_contexts), self.MAX_CONTEXT_LEN))
        self.y_data_test = np.zeros((len(test_values), self.MAX_VALUE_LEN))

        for i in range(len(train_contexts)):
            enumerated_context = np.array(train_contexts[i])
            enumerated_value = np.array(train_values[i])
            np.copyto(self.X_data_train[i, :len(enumerated_context)], enumerated_context)
            np.copyto(self.y_data_train[i, :len(enumerated_value)], enumerated_value)

        for i in range(len(test_contexts)):
            enumerated_context = np.array(test_contexts[i])
            enumerated_value = np.array(test_values[i])
            np.copyto(self.X_data_test[i, :len(enumerated_context)], enumerated_context)
            np.copyto(self.y_data_test[i, :len(enumerated_value)], enumerated_value)
