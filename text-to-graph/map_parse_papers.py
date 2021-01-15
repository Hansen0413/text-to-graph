import sys
import csv
import time
from nltk.tokenize import word_tokenize
from scipy.sparse import dok_matrix, save_npz


def normalize(sentence):
    # normalize abstract, make it easier to find keys in abstract
    words = word_tokenize(sentence)
    n_sentence = ' '.join(words).lower()
    return ' ' + n_sentence + ' '


def read_key_list(filename):
    # generate list of keys from csv document
    with open(filename) as fin:
        reader = csv.reader(fin)
        result = list(reader)
    key_list = []
    for line in result:
        key_list.append(line[1])
    return key_list


def key_in_abstract(key, abstract):
    # judge if key is in abstract
    key = ' ' + key + ' '
    return key in abstract


def update_graph_sparse(graph, sentence, key_list):
    # update graph according to one abstract
    tmp = []
    for i, key in enumerate(key_list):
        if key_in_abstract(key, sentence):
            tmp.append(i)
            for j in tmp:
                graph[i, j] += 1
                graph[j, i] += 1
            graph[i, i] -= 1


def parse_abstracts(filename, key_list):
    # generate the graph
    graph = dok_matrix((len(key_list), len(key_list)))

    with open(filename) as fin:
        i = 1
        for line in fin.readlines():
            if i % 100 == 0:
                print('Processing line: {}'.format(i))
            update_graph_sparse(graph, line, key_list)
            i += 1

    return graph


def main(filenames):
    key_list = read_key_list(filenames[1])

    graph = parse_abstracts(filenames[0], key_list)
    # save the graph
    save_npz('graph_{}'.format(filenames[0].split('/')[-1]), graph.tocoo()) 


if __name__ == '__main__':
    if len(sys.argv) > 2:
        start = time.perf_counter()

        main(sys.argv[1:])

        elapsed = (time.perf_counter() - start)
        print('time: {:.2f}'.format(elapsed))
    else:
        print('Usage: map_parse_papers.py <filename> <key list file>')
