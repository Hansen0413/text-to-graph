import sys
import time
import glob
from scipy.sparse import coo_matrix, load_npz, save_npz
from sklearn.preprocessing import normalize


def sum_matrix(files) -> coo_matrix:
    # merge graphs generated from different parts of kaggle data
    result = None
    for file in glob.glob(files):
        part_matrix = load_npz(file).tocoo()
        if result is not None:
            result += part_matrix
        else:
            result = part_matrix
        print('The file {} is processed'.format(file))
    return result


def cal_pmi(matrix) -> coo_matrix:
    # change coefficients in graph to PMI
    diag = matrix.diagonal()
    pmi_matrix = matrix.tocsr()

    print('Divide the matrix by the diagonal...')
    for i in range(matrix.get_shape()[0]):
        if diag[i] > 0:
            pmi_matrix[i] /= diag[i]
        if i % 100 == 0:
            print('Dividing at line {}'.format(i))

    print('Transpose the matrix and divide it by the raw diagonal...')
    pmi_matrix = pmi_matrix.transpose()
    for i in range(matrix.get_shape()[0]):
        if diag[i] > 0:
            pmi_matrix[i] /= diag[i]
        if i % 100 == 0:
            print('Dividing at line {}'.format(i))

    pmi_matrix.setdiag(0)
    # normalize every row in order to do random walk
    pmi_matrix = normalize(pmi_matrix, norm='l1')
    return pmi_matrix.tocoo()


def main(argv):
    print('Begin...')
    matrix = sum_matrix(argv[0])
    print('Combined all partial matrix into one')
    pmi_matrix = cal_pmi(matrix)
    print('Calculated PMI')
    save_npz(argv[1], pmi_matrix)
    print('Save the result into file')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        start = time.perf_counter()

        main(sys.argv[1:])

        elapsed = (time.perf_counter() - start)
        print('time: {:.2f}'.format(elapsed))
    else:
        print('Usage: reduce_parse_results.py <file filter> <output file>')
