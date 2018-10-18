# import dataloader
import numpy as np

class EntriesProcessor:

    def __init__(self, max_value_len=20, max_context_len=20):
        self.uniq_symbols = set()
        self.symbols_dict = {'<PAD>': 0, '<EOS>' : 1, '<UNK>': 2 }
        self.symbols_dict_rev = {0: '<PAD>', 1: '<EOS>', 2 : '<UNK>'}
        self.symbols_counter = 3
        self.X_data = None
        self.y_data = None
        self.MAX_VALUE_LEN = max_value_len
        self.MAX_CONTEXT_LEN = max_context_len

    def process(self, entries):
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

        contexts = []
        values = []
        for entry in entries:
            if len(entry.value) <= self.MAX_VALUE_LEN - 1 and len(entry.context) <= self.MAX_CONTEXT_LEN - 1:
                contexts.append(list(map(lambda x: self.symbols_dict[x], entry.context.lower())) + [self.symbols_dict['<EOS>']])
                values.append(list(map(lambda x: self.symbols_dict[x], entry.value.lower()))+ [self.symbols_dict['<EOS>']])
        assert len(contexts) == len(values)

        self.X_data = np.zeros((len(contexts), self.MAX_CONTEXT_LEN))
        self.y_data = np.zeros((len(values), self.MAX_VALUE_LEN))

        for i in range(len(contexts)):
            enumerated_context = np.array(contexts[i])
            enumerated_value = np.array(values[i])
            np.copyto(self.X_data[i, :len(enumerated_context)], enumerated_context)
            np.copyto(self.y_data[i, :len(enumerated_value)], enumerated_value)