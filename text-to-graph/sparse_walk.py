import sys
import csv
import numpy as np
from scipy.sparse import *


def read_key_list(filename):
    with open(filename) as fin:
        reader = csv.reader(fin)
        result = list(reader)
    key_list = []
    for line in result:
        key_list.append(line[1])
    return key_list


def generate_sentences(graph, v, l, key_list, fout):
    # random walk in the graph
    sums = graph.sum(axis=1)
    for i in range(graph.shape[0]):
        if sums[i, 0] == 0: continue
        for _ in range(v):
            tmp = [i]
            sentences = key_list[i]
            index = i
            for j in range(l - 1):
                index = np.random.choice(graph.rows[index], p=graph.data[index])
                tmp.append(index)
                sentences += ', ' + key_list[index]
            print(sentences)
            save_sentence(fout, tmp)


def save_sentence(fout, sentence):
    for num in sentence:
        fout.write(str(num) + ' ')
    fout.write('\n')


def main(sentences_file, graph_file, key_file, v=80, l=10):
    print('Reading graph...')
    graph = load_npz(graph_file).tolil()
    key_list = read_key_list(key_file)

    print('Generating sentences...')
    with open(sentences_file, 'w') as f_output:
        generate_sentences(graph, v, l, key_list, f_output)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        if len(sys.argv) > 5:
            main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        else:
            main(sys.argv[1], sys.argv[2], sys.argv[3])

    else:
        print('Usage: sparse_walk.py <sentences output filename> <graph filename> <key filename> [v l]')

