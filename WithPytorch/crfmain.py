from entriesprocessor import EntriesProcessor
from trainerpytorchcrf import TrainerPythonCRF
from model import Encoder, DecoderPythonCRF
from mockentries import two_case_entries

import torch

entries = two_case_entries(100)
ep = EntriesProcessor(20, 40, 0)
# voc_size = 10
ep.process(entries)
voc_size = ep.symbols_counter
print(len(ep.X_data_test), len(ep.X_data_train))
EMBEDDING_SIZE = 10
HIDDEN_SIZE = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_size=voc_size, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE).to(device)
decoder = DecoderPythonCRF(hidden_size=HIDDEN_SIZE,
                           embedding_size=EMBEDDING_SIZE,
                           output_size=voc_size,
                           max_length=40).to(device)
trainer = TrainerPythonCRF(encoder, decoder, ep, max_input_length=40, max_output_length=20, device=device)
trainer.train(1, batch_size=10)
# a,b,c = trainer.evaluate_with_attn(ep.X_data_train[1:2])
print('kek')
ethalons, results, inputs, matched, distances = trainer.test_model(False)
# print(np.mean(matched))
# print(np.mean(distances))
# print(inputs[:10])