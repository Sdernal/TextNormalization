from os.path import join
from os import listdir
import xml.etree.ElementTree as ET
import re

def distance(val1, val2):
    "Calculates the Levenshtein distance between a and b."
    a, b = val1, val2
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current_row = range(n+1) # Keep current and previous row, not entire matrix
    for i in range(1, m+1):
        previous_row, current_row = current_row, [i]+[0]*n
        for j in range(1,n+1):
            add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]
            if a[j-1] != b[i-1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]

basicentity_label = '{http://www.abbyy.com/ns/BasicEntity#}'
rdf_label = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
rdfs_label = '{http://www.w3.org/2000/01/rdf-schema#}'
aux_label = '{http://www.abbyy.com/ns/Aux#}'

class Entry:
    def __init__(self):
        self.offset = None
        self.length = None
        self.value = None
        self.context = None
        self.context_offset = None

    def __str__(self):
        return 'Offset:{0}\tLength:{1}\tValue:{2}\tContext:{3}\tContext offset:{4}'.format(
            self.offset, self.length, self.value, self.context, self.context_offset)

class DataLoader:
    def __init__(self, window_size: int = 0):
        self.entries = []
        self.window_size = window_size

    def parse_person_corpus(self, data_dir):
        for data_file in listdir(data_dir):
            data_file_path = join(data_dir, data_file)
            self.entries.extend(self.parse_entries(data_file_path))

    def parse_rdf_corpus(self, data_dir):
        for rdf in listdir(data_dir):
            rdf_path = join(data_dir, rdf)
            self.entries.extend(self.parse_rdf(rdf_path))

    def parse_entries(self, dir_path):
        entries = DataLoader.parse_markup(join(dir_path, 'anno.markup.xml'))
        text = DataLoader.parse_text(join(dir_path, 'text.txt'))
        for entry in entries:
            context_start = max(0, entry.offset - self.window_size)
            context_end = min(len(text), entry.offset + entry.length + self.window_size)
            entry.context = text[context_start:context_end]
            entry.context_offset = context_start
        return entries
    
    @staticmethod
    def parse_markup(file_path):
        root = ET.parse(file_path).getroot()
        entries = []
        for entry_node in root.findall('entry'):
            entry = Entry()
            entry.offset = int(entry_node.find('offset').text)
            entry.length = int(entry_node.find('length').text)
            entry.value = entry_node.find('attribute').find('value').text
            entries.append(entry)
        return entries

    @staticmethod
    def parse_text(file_path):
        text = open(file_path, 'r', encoding='windows-1251').read()
        text = re.sub('\n', '\n\r', text)
        return text

    def parse_rdf(self, file_path):
        entries = []
        root = ET.parse(file_path).getroot()
        person_nodes = root.findall(basicentity_label + 'Person')

        person_labels = []
        person_ids = []

        for person_node in person_nodes:
            try:
                label = person_node.find(rdfs_label + 'label').text
                node_id = person_node.get(rdf_label + 'nodeID')
                person_labels.append(label)
                person_ids.append(node_id)
            except Exception as e:
                print(e, file_path)

        annotations_node = root.find(aux_label + 'TextAnnotations')
        text = annotations_node.find(aux_label + 'document_text').text

        annotations = annotations_node.findall(aux_label + 'annotation')
        for annotation in annotations:
            instance_node = annotation.find(aux_label + 'InstanceAnnotation')
            node_id = instance_node.find(aux_label + 'instance').get(rdf_label + 'nodeID')
            if node_id not in person_ids:
                continue

            annotation_start = int(instance_node.find(aux_label + 'annotation_start').text)
            annotation_end = int(instance_node.find(aux_label + 'annotation_end').text)
            direct_context = text[annotation_start:annotation_end]
            value = person_labels[person_ids.index(node_id)]

            if distance(direct_context, value) > len(value) // 2:
                continue

            context_start = max(0, annotation_start - self.window_size)
            context_end = min(len(text), annotation_end + self.window_size)
            wide_context = text[context_start:context_end]

            entry = Entry()
            entry.value = value
            entry.context = wide_context
            entry.offset = annotation_start
            entry.context_offset = context_start
            entry.length = annotation_end - annotation_start

            entries.append(entry)
        return entries