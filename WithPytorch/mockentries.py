from dataloader import Entry
import random

def generate_entries(count, max_len=10):
    # voc = ['a','b','c','d','e','f','g','h','i','j','k','l','m','o','p','q','r','s','t','u','v','w','x','y','z']
    voc = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    entries = []
    for _ in range(count):
        word = [voc[random.randint(0, len(voc) - 1)] for _ in range(max_len)]
        word = ''.join(word)
        result = word + 'a'
        entry = Entry()
        entry.context = word
        entry.value = result
        entries.append(entry)

    return entries