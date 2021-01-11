import pickle
import bcolz
import numpy as np

## Pellaro, 2018 https://gist.github.com/martinpella/39fa038d9b18c2794862a16547e935ec#file-glove_dict-py
## creates and stores 300 dim glove vectors at given glove_path
glove_path = 'utils/glove.6B'
def initialize_glove():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.300.dat', mode='w')

    with open(f'{glove_path}/glove.6B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=f'{glove_path}/6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.300_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.300_idx.pkl', 'wb'))

## creates dictionary for all vectors indexed by word string
def wordvecDict():
    vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove, glove_path
## end Pellaro

## Questions must be 14 x 300
def parse_questions():
    return


## Images should be K x 2048
def parse_images():
    return