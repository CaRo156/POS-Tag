import re, os
from utils import *

def get_sentence_from_pos(file):
    with open(file, 'r') as f:
        s = f.read()
        s = s.replace('``/``', '').replace('}/)', '').replace(')/)', '').replace('{/(', '')\
            .replace('(/(', '').replace("''/''", '').replace('`/``', '').replace("'/''", '')
        s = re.sub('[][=]', '', s).strip()
        words = s.split()
        for i in range(len(words)):
            if words[i] == './.' or words[i] == '!/.' or words[i] == '?/.':
                words[i] += '\n'
    return ' '.join(words)

def pos_to_txt(des=TAGGED_SENTENCES):
    sentences = []
    for folder in os.listdir(DATA_DIR):
        folder_path = DATA_DIR + folder + '/'
        for file in os.listdir(folder_path):
            sentences.append(get_sentence_from_pos(folder_path + file).replace('\n ', '\n'))
        with open(des, 'w') as des_file:
            des_file.write(''.join(sentences))

def tag_filter(source=TAGGED_SENTENCES, des=RAW_SENTENCES, tag_file=TAGS):
    with open(source, 'r') as f:
        sentences = []
        sentences_tag = []
        for line in f:
            words = line.split(' ')
            if len(words) < 2:
                continue
            new_line = []
            line_tag = []
            for w in words[:-1]:
                if w.find('/') < 0:
                    print (w)
                    continue
                i = w.rindex('/')
                new_line.append(w[:i])
                line_tag.append(w[i + 1 :])
            sentences.append(' '.join(new_line))
            sentences_tag.append(' '.join(line_tag))
    with open(des, 'w') as f:
        f.write('\n'.join(sentences))
    with open(tag_file, 'w') as f:
        f.write('\n'.join(sentences_tag))

def anomaly_word():
    with open(RAW_SENTENCES, 'r') as f:
        for line in f:
            words = line.lower().strip().split(' ')
            for w in words:
                if w == ',' or w == '.' or w == '&' or w == '%' or w == ';' or w == '--' or w == '?' or w == '!':
                    continue
                if w .replace(',', '').replace('.', '').isdigit():
                    continue
                for c in w:
                    if 97 > ord(c) or ord(c) > 122:
                        print (w)
                        break

if __name__ == '__main__':
    #pos_to_txt()
    #tag_filter()
    #anomaly_word()
    print()
