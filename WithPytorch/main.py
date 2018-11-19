from dataloader import DataLoader
from os.path import abspath
from entriesprocessor import EntriesProcessor
from trainer import Trainer
from model import Decoder, Encoder
import torch
import numpy as np

EMBEDDING_SIZE = 10
HIDDEN_SIZE = 200
print(abspath('./'))
loader = DataLoader(10)
loader.parse_person_corpus(abspath('../Persons-1000/collection'))
loader.parse_rdf_corpus(abspath('../../corpus_for_pakhomov_2'))

ep = EntriesProcessor(20,40,10)
ep.process(loader.entries)
voc_size = ep.symbols_counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_size=voc_size, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE).to(device)
decoder = Decoder(hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, output_size=voc_size, max_length=40).to(device)
trainer = Trainer(encoder,decoder,ep,max_input_length=40, max_output_length=20)

trainer.train(1, batch_size=256)
ethalons, results, inputs, matched, distances = trainer.test_model(False)
print(np.mean(matched))
print(np.mean(distances))
print(inputs[:10])