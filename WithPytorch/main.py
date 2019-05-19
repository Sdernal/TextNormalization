from dataloader import DataLoader, distance
from os.path import abspath
from entriesprocessor import EntriesProcessor
from trainer import Trainer
from trainerpytorchcrf import TrainerPythonCRF
from model import Decoder, Encoder, DecoderPythonCRF
import torch
import re
import numpy as np
from mockentries import two_case_entries


entries = two_case_entries(1000)
ep = EntriesProcessor(20, 40, 0)
ep.process(entries)
voc_size = ep.symbols_counter

EMBEDDING_SIZE = 10
HIDDEN_SIZE = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_size=voc_size, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE).to(device)
decoder = Decoder(hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, output_size=voc_size, max_length=40).to(device)
trainer = Trainer(encoder,decoder,ep,max_input_length=40, max_output_length=20)
trainer.train(10)
exit(0)
def words(text):
    wds = filter(lambda x: len(x) > 0, re.split('\.|-| ', text))
    return list(wds)


def delete_excess(text, value):
    text_words = text.split()
    value_words = value.split()

    result = []
    for t in text_words:
        for v in value_words:
            if distance(t, v) < 3:
                result.append(v)
    if len(text_words) == len(result):
        #         print('\t', text,'\t', ' '.join(result), '\t', value)
        return ' '.join(result)

EMBEDDING_SIZE = 10
HIDDEN_SIZE = 200
print(abspath('./'))
loader = DataLoader(10)
loader.parse_person_corpus(abspath('../Persons-1000/collection'))
loader.parse_rdf_corpus(abspath('../../corpus_for_pakhomov_2'))

bad_entries = []
edited_enties = []
for entry in loader.entries:
    entry_start = entry.offset - entry.context_offset
    entry_end = entry_start + entry.length
    entry_text = entry.context[entry_start:entry_end]
    diff_len = abs(len(words(entry_text)) - len(words(entry.value)))
    if diff_len > 0:
        try_to_fix = delete_excess(entry_text, entry.value)
        if try_to_fix is None:
            bad_entries.append(entry)
        else:
            entry.value = try_to_fix
#             print(entry)
            edited_enties.append(entry)
    else:
        edited_enties.append(entry)

ep = EntriesProcessor(20,40,10)
ep.process(edited_enties)
voc_size = ep.symbols_counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_size=voc_size, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE).to(device)
# decoder = Decoder(hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, output_size=voc_size, max_length=40).to(device)
decoder = DecoderPythonCRF(hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, )
# trainer = Trainer(encoder,decoder,ep,max_input_length=40, max_output_length=20)
trainer = TrainerPythonCRF(encoder, )
trainer.train(1, batch_size=256)
a,b,c = trainer.evaluate_with_attn(ep.X_data_train[1:2])
print('kek')
# ethalons, results, inputs, matched, distances = trainer.test_model(False)
# print(np.mean(matched))
# print(np.mean(distances))
# print(inputs[:10])