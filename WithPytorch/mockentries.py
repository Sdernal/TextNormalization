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


def two_case_entries(count, max_len=7):

    BASE_VOC = ['b','c','d','e','f','g','h']

    entries = []
    for _ in range(count):
        base_adj_len = random.randint(3, max_len)
        base_sub_len = random.randint(3, max_len)
        base_verb_len = random.randint(3, max_len)

        gender = random.randint(0,1)
        case = random.randint(0,1)

        base_adj = [BASE_VOC[random.randint(0,len(BASE_VOC) - 1)] for _ in range(base_adj_len)]
        base_adj = ''.join(base_adj)
        base_sub = [BASE_VOC[random.randint(0, len(BASE_VOC) - 1)] for _ in range(base_sub_len)]
        base_sub = ''.join(base_sub)
        base_verb = [BASE_VOC[random.randint(0,len(BASE_VOC) - 1)] for _ in range(base_verb_len)]
        base_verb = ''.join(base_verb)

        adj_gen = base_adj + 'a' if gender == 0 else base_adj + 'ou'
        adj_nom = base_adj if gender == 0 else base_adj + 'a'

        sub_gen = base_sub + 'a' if gender == 0 else base_sub + 'u'
        sub_nom = base_sub if gender == 0 else base_sub + 'a'

        if case == 0:
            verb = base_verb if gender == 0 else base_verb + 'a'
        else:
            verb = base_verb + 'u'

        entry = Entry()
        entry.value = ' '.join([adj_nom, sub_nom])
        entry.context = ' '.join([adj_gen, sub_gen, verb]) if case !=0 else ' '.join([entry.value, verb])
        entry.context_offset = 0
        entry.offset = 0
        entry.length = len(entry.context)
        entries.append(entry)

    return entries