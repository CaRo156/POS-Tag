from utils import *
from math import log
from random import shuffle
import time

def get_train_test_data(split=0.1):
    with open(RAW_SENTENCES, 'r') as f:
        sentences = f.read().split('\n')
    with open(TAGS, 'r') as f:
        tags = f.read().split('\n')
    idx_shuff = [i for i in range(len(sentences))]
    shuffle(idx_shuff)
    sentences_shuff = [sentences[i] for i in idx_shuff]
    tags_shuff = [tags[i] for i in idx_shuff]
    split_idx = int(len(sentences) * (1 - split))

    return sentences_shuff[:split_idx], tags_shuff[:split_idx], sentences_shuff[split_idx:], tags_shuff[split_idx:]

class HMM(object):

    def __init__(self):
        self.M = 0
        self.unknown_words = {}
        self.tags_count = {}
        self.tags = []
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.suffix = {}
        self.weights = [0, 0, 0]
        self.B = {}

    def train(self, X_train, Y_train):
        self.M = len(Y_train)
        self.train_unigram(Y_train)
        self.train_bigram(Y_train)
        self.train_trigram(Y_train)
        self.train_suffiex(X_train, Y_train)
        self.deleted_interpolation()
        self.normalize()
        self.get_words_tag_pr(X_train, Y_train)
    
    def train_unigram(self, Y_train):
        for y in Y_train:
            if len(y) < 1:
                continue
            tags = y.strip().split(' ')
            for tag in tags:
                if tag in self.tags_count.keys():
                    self.tags_count[tag] += 1
                else:
                    self.tags_count[tag] = 1
        self.unigram = self.tags_count
        self.tags = [k for k in self.tags_count.keys()]

    def train_bigram(self, Y_train):
        self.bigram = {k: {} for k in self.tags + ['*']}
        for k in self.bigram.keys():
            self.bigram[k] = {ki: 0 for ki in self.tags + ['*', 'STOP']}

        for y in Y_train:
            tags = ['*', '*'] + y.strip().split(' ')
            for i in range(len(tags) - 1):
                self.bigram[tags[i]][tags[i + 1]] += 1
            self.bigram[tags[-1]]['STOP'] += 1

    def train_trigram(self, Y_train):
        self.trigram = {k: {} for k in self.tags + ['*']}
        for dict in self.trigram.values():
            for k in self.tags + ['*']:
                dict[k] = {}
                dict[k] = {ki: 0 for ki in self.tags + ['*', 'STOP']}

        for y in Y_train:
            tags = ['*', '*'] + y.strip().split(' ')
            for i in range(len(tags) - 2):
                self.trigram[tags[i]][tags[i + 1]][tags[i + 2]] += 1
            self.trigram[tags[-2]][tags[-1]]['STOP'] += 1

    def train_suffiex(self, X_train, Y_train):
        open_class = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD',
                      'VBZ', 'VBG', 'JJ', 'VBN', 'RB', 'RBR', 'RBS']
        for x, y in zip(X_train, Y_train):
            words = x.split(' ')
            tags = y.split(' ')
            for w, t in zip(words, tags):
                if t not in open_class or w.replace(',', '').replace('.', '').isdigit() or len(w) < 5:
                    continue
                length = len(w)
                max_length = min(len(w) - 2, 4)
                for i in range(length - 1, length - max_length, -1):
                    if w[i:] not in self.suffix:
                        self.suffix[w[i:]] = {k : 0 for k in self.tags}
                    self.suffix[w[i:]][t] += 1
        for s in self.suffix:
            self.suffix[s] = {t : self.suffix[s][t] / self.tags_count[t] for t in self.tags}

    def normalize(self):
        all_tag_counts = sum(self.tags_count.values())
        self.unigram = {k: self.tags_count[k] / all_tag_counts for k in self.tags}

        for k1 in self.trigram.keys():
            for k2 in self.trigram.keys():
                for k3 in self.tags:
                    if self.bigram[k1][k2] == 0:
                        continue
                    self.trigram[k1][k2][k3] /= self.bigram[k1][k2]

        for k, dict in self.bigram.items():
            for t in dict:
                if t == 'STOP':
                    dict[t] = (dict[t] + 1) / (self.M + len(self.tags))
                elif t == '*':
                    dict[t] = 1
                else:
                    dict[t] = (dict[t] + 1) / (self.tags_count[t] + len(self.tags))

    def deleted_interpolation(self):
        N = sum(self.tags_count.values())
        for k1 in self.trigram.keys():
            for k2 in self.trigram.keys():
                for k3 in self.tags:
                    if self.trigram[k1][k2][k3] > 0:
                        if self.bigram[k1][k2] != 1:
                            c3 = (self.trigram[k1][k2][k3] - 1) / (self.bigram[k1][k2] - 1)
                        else:
                            c3 = 0
                        if k2 == '*':
                            c2 = (self.bigram[k2][k3] - 1) / (self.M - 1)
                        elif self.unigram[k2] != 1:
                            c2 = (self.bigram[k2][k3] - 1) / (self.unigram[k2] - 1)
                        else:
                            c2 = 0
                        c1 = (self.unigram[k3] - 1) / (N - 1)
                        imax = [c1, c2, c3].index(max([c1, c2, c3]))
                        self.weights[imax] += self.trigram[k1][k2][k3]

        sum_weight = sum(self.weights)
        self.weights = [i / sum_weight for i in self.weights]

    def cal_trigram(self, w, u, v):
        #return self.trigram[w][u][v]
        if v == 'STOP':
            return self.weights[1] * self.bigram[u][v] + \
                   self.weights[2] * self.trigram[w][u][v]

        return self.weights[0] * self.unigram[v] + \
               self.weights[1] * self.bigram[u][v] + \
               self.weights[2] * self.trigram[w][u][v]

    def get_words_tag_pr(self, sentences, sentences_tag):
        for s, t in zip(sentences, sentences_tag):
            words = s.strip().split(' ')
            tags = t.strip().split(' ')
            for w, t in zip(words, tags):
                if w.replace(',', '').replace('.', '').isdigit():
                    continue
                if w in self.B.keys():
                    self.B[w][t] += 1
                else:
                    self.B[w] = {k: 0 for k in self.tags}
                    self.B[w][t] = 1

        for dict in self.B.values():
            #total = sum(dict.values())
            for k in dict.keys():
               dict[k] = (dict[k]) / (self.tags_count[k])
               #dict[k] /= total

    def tag_unknown_word(self, word):
        tags = {k: 0 for k in self.tags}
        if word.replace(',', '').replace('.', '').isdigit():
            tags['CD'] = 1
            return tags
        word_suffix = [word[i:] for i in range(-4, 0)]
        for s in word_suffix:
            if s in self.suffix:
                return self.suffix[s]

        if word.replace(',', '').replace('.', '').isdigit():
            tags['CD'] = 1
        elif len(word) == 1:
            tags['SYM'] = 1
        elif word.endswith('able') or word.find('-') > 0 or word.endswith('al') or word.endswith('ous'):
            tags['JJ'] = 1
        elif word.endswith('ed'):
            tags['VBD'] = 1
        elif word.endswith('ing'):
            tags['VBG'] = 1
        elif ord(word[0]) > 90:
            if word.endswith('s'):
                tags['NNS'] = 1
            else:
                tags['NN'] = 1
        else:
            if word.endswith('s') and word.lower()[:-1] in self.B.keys():
                tags['NNPS'] = 1
            else:
                tags['NNP'] = 1

        return tags

    def viterbi_bigram(self, X):
        words = X.split(' ')
        if words[0] not in self.B.keys() and words[0].lower() in self.B.keys():
            words[0] = words[0].lower()
        for w in words:
            if w not in self.B.keys():
                self.unknown_words[w] = 1
                self.B[w] = self.tag_unknown_word(w)
            elif w in self.unknown_words.keys():
                self.unknown_words[w] += 1
        V = [{}]
        for st in self.tags:
            V[0][st] = {'pr': (self.bigram['*'][st]) * (self.B[words[0]][st]), 'prev': None}

        for i in range(1, len(words)):
            V.append({})
            for st in self.tags:
                tr_pr = [V[i - 1][prev]['pr'] * self.bigram[prev][st]
                            for prev in self.tags]
                max_pr = max(tr_pr)
                imax = tr_pr.index(max_pr)
                pr = max_pr * (self.B[words[i]][st])
                V[i][st] = {'pr': pr, 'prev': self.tags[imax]}

        opt_path = []
        max_pr = max(v['pr'] for v in V[-1].values())
        prev = None
        for st, dict in V[-1].items():
            if dict['pr'] == max_pr:
                opt_path.append(st)
                prev = st
                break
        for i in range(len(V) - 1, 0, -1):
            opt_path.insert(0, V[i][prev]['prev'])
            prev = V[i][prev]['prev']
        return opt_path

    def viterbi_trigram(self, X):
        words = X.split(' ')
        N = len(words)
        if words[0].lower() in self.B.keys():
            words[0] = words[0].lower()
        for w in words:
            if w not in self.B.keys():
                self.unknown_words[w] = 1
                self.B[w] = self.tag_unknown_word(w)
            elif w in self.unknown_words.keys():
                self.unknown_words[w] += 1
        words = [''] + words
        def K(k):
            if k in (-1, 0):
                return ['*']
            else:
                return self.tags

        def argmax(ls):
            return max(ls, key=lambda x: x[1])

        V = {}
        V[0, '*', '*'] = 1.
        bp = {}

        for k in range(1, N + 1):
            for u in K(k - 1):
                for v in K(k):
                    bp[k, u, v], V[k, u, v] = argmax(
                        [(w, V[k - 1, w, u] * self.cal_trigram(w, u, v) * self.B[words[k]][v]) for w in K(k - 2)])
        y = [''] * (N + 1)
        (y[N - 1], y[N]), score = argmax([((u, v), V[N, u, v] * self.cal_trigram(u, v, 'STOP')) for u in K(N - 1) for v in K(N)])
        for k in range(N - 2, 0, -1):
            y[k] = bp[k + 2, y[k + 1], y[k + 2]]
        y[0] = '*'
        return y[1:]

    def accuracy_score(self, y_true, y_predict):
        count = sum([1 if y_predict[i] == y_true[i] else 0 for i in range(len(y_true))])
        return round(float(count) / len(y_true), 3)

    def precision(self, y_true, y_predict, c):
        tp = sum([1 if y_predict[i] == y_true[i] and y_predict[i] == c else 0 for i in range(len(y_true))])
        fp = sum([1 if y_predict[i] != y_true[i] and y_predict[i] == c else 0 for i in range(len(y_true))])
        return round(float(tp) / (tp + fp), 3)


    def recall(self, y_true, y_predict, c):
        tp = sum([1 if y_predict[i] == y_true[i] and y_predict[i] == c else 0 for i in range(len(y_true))])
        fn = sum([1 if y_predict[i] != y_true[i] and y_true[i] == c else 0 for i in range(len(y_true))])
        return round(float(tp) / (tp + fn), 3)

    def f1_score(self, p, r):
        return round(float(2 * p * r) / (p + r), 3)

    def test(self, X_test, Y_test, log=False):
        predict_sentences_tag = []
        y = []
        y_p = []
        f1_tags = {}
        for yt in Y_test:
            y += yt.split(' ')
        print ('Number of sentences: ', len(Y_test))
        print ('Number of words: ', len(y))
    
        for i, x in enumerate(X_test):
            result = self.viterbi_bigram(x)
            predict_sentences_tag.append(' '.join(result))
    
        known_word_acc = 0
        unknown_word_acc = 0
        total_unknown_word = sum(self.unknown_words.values())
        for x, p, yi in zip(X_test, predict_sentences_tag, Y_test):
            words = x.split(' ')
            p_tags = p.split(' ')
            tags = yi.split(' ')
            y_p += p_tags
            diff = len(tags) - len(p_tags)
            for i in range(diff):
                y_p += ['UNK']
            for w, t1, t2 in zip(words, p_tags, tags):
                if w in self.unknown_words.keys() and t1 == t2:
                    unknown_word_acc += 1
                elif t1 == t2:
                    known_word_acc += 1

        accuracy = round((known_word_acc + unknown_word_acc )/ len(y), 4)
        known_word_acc = round(known_word_acc / (len(y) - total_unknown_word), 4)
        unknown_word_acc = round(unknown_word_acc / total_unknown_word, 4)
        print ('Tag word accuracy: ', accuracy)
        print ('Known word tag accuracy: ', known_word_acc)
        print ('Unknown words: ', total_unknown_word)
        print ('Unknown words tag accuracy: ', unknown_word_acc)
        for tag in self.tags:
            if tag == ',' or tag == ':' or tag == '$' or tag.find('|') > 0:
                continue
            f1_tags[tag] = {}
            try:
                pr = self.precision(y, y_p, tag)
            except ZeroDivisionError:
                pr = 0
            try:
                rc = self.recall(y, y_p, tag)
            except ZeroDivisionError:
                rc = 0
            try:
                f1 = self.f1_score(pr, rc)
            except ZeroDivisionError:
                f1 = 0
            f1_tags[tag]['P'] = pr
            f1_tags[tag]['R'] = rc
            f1_tags[tag]['F'] = f1

        if not log:
            return accuracy, known_word_acc, unknown_word_acc, f1_tags
        with open(LOG_DIR + 'log.txt', 'w') as f:
            for x, p, t in zip(X_test, predict_sentences_tag, Y_test):
                words = x.split(' ')
                p_tags = p.split(' ')
                tags = t.split(' ')
                for i, w in enumerate(words):
                    if p_tags[i] != tags[i]:
                        if w in self.unknown_words.keys():
                            f.write('<' + w + '/' + p_tags[i] + '(' + tags[i] + ')> ')
                        else:
                            f.write(w + '/' + p_tags[i] + '(' + tags[i] + ') ')
                    else:
                        f.write(w + '/' + p_tags[i] + ' ')
                f.write('\n-------------------------------------------------\n')
        return accuracy, known_word_acc, unknown_word_acc, f1_tags

def cross_validate(tests=10, test_size=0.1):
    t0 = time.time()
    acc = 0
    kw_acc = 0
    unk_acc = 0
    f1_all_tests = {}
    print('Number of tests: ', tests)
    for i in range(tests):
        print('--------- [TEST %d] ---------' % (i + 1))
        X_train, Y_train, X_test, Y_test = get_train_test_data(test_size)
        hmm = HMM()
        hmm.train(X_train, Y_train)
        a, k, u, f1 = hmm.test(X_test, Y_test, log=False)
        acc += a
        kw_acc += k
        unk_acc += u
        if i == 0:
            f1_all_tests = f1
        else:
            for tag in f1_all_tests.keys():
                for s in f1_all_tests[tag].keys():
                    if s not in f1[tag]:
                        f1[tag][s] = 0
                    f1_all_tests[tag][s] += f1[tag][s]
    acc = round(acc / tests, 4)
    kw_acc = round(kw_acc / tests, 4)
    unk_acc = round(unk_acc / tests, 4)
    print ('--------- [AVERAGE] ---------')
    print('Accuracy score: ', acc)
    print('Known word tag accuracy: ', kw_acc)
    print('Unknown word tag accuracy: ', unk_acc)
    print ('F1-Score: ')
    for tag in f1_all_tests.keys():
        print ('Tag: ', tag)
        print ('Precision: ', round(f1_all_tests[tag]['P'] / tests, 3))
        print('Recall: ', round(f1_all_tests[tag]['R'] / tests, 3))
        print('F1: ', round(f1_all_tests[tag]['F'] / tests, 3))

    print('Time: ', time.time() - t0)

if __name__ == '__main__':
    cross_validate(tests=10, test_size=0.01)
