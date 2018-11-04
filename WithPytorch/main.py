from dataloader import DataLoader
from os.path import abspath
from entriesprocessor import EntriesProcessor
from trainer import Trainer
from model import Decoder, Encoder
import torch
from mockentries import generate_entries

EMBEDDING_SIZE = 10
HIDDEN_SIZE = 100
# print(abspath('./'))
# loader = DataLoader(10)
# loader.parse_person_corpus(abspath('../Persons-1000/collection'))
# loader.parse_rdf_corpus(abspath('../../corpus_for_pakhomov_2'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ep = EntriesProcessor(15,15)
# voc_size = 10
entries = generate_entries(10000)
ep.process(entries)
voc_size = ep.symbols_counter
encoder = Encoder(input_size=voc_size, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE).to(device)
decoder = Decoder(hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, output_size=voc_size, max_length=40).to(device)
trainer = Trainer(encoder,decoder,ep)
trainer.train(20)
# trainer.test_training()