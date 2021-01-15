import gensim
import sys

def main(sentences_file, vector_file):
    sentences = gensim.models.word2vec.Text8Corpus(sentences_file)
    model = gensim.models.Word2Vec(sentences, sg=1, min_count=0)
    model.wv.save_word2vec_format(vector_file)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Usage: word2vec.py <sentences input filename> <vectors output filename>')