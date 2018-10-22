from dataloader import DataLoader
from os.path import abspath
from entriesprocessor import EntriesProcessor
from trainer import Trainer
from model import Decoder, Encoder

EMBEDDING_SIZE = 10
HIDDEN_SIZE = 100

# loader = DataLoader(10)
# loader.parse_person_corpus('../Person-1000/collection')
# loader.parse_rdf_corpus('../../corpus_for_pakhomov_2')

ep = EntriesProcessor(20,40)
# ep.process(loader.entries)
voc_size = 10
encoder = Encoder(input_size=voc_size, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE)
decoder = Decoder(hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, output_size=voc_size, max_length=40)
trainer = Trainer(encoder,decoder,ep)
trainer.test_training()