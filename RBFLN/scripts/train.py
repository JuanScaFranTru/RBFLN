"""Train an rbfln model

Usage:
  train.py -m <n> -n <n> -o <file>
  train.py -h | --help

Options:
  -m <n>        Number of neurons in the hidden layer.
  -n <n>        Number of iterations.
  -o <file>     Output file
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from RBFLN.rbfln import RBFLN

if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the data
    Q = 115
    xs = [[float(x)] for x in range(Q)]
    ts = [[float(x**2)] for x in range(Q)]
    N = 1

    # Read the number of neurons in the hidden layer
    M = int(opts['-m'])

    # Read the number of iterations
    niter = int(opts['-n'])

    model = RBFLN(xs, ts, M, N, niter)

    # save it
    filename = opts['-o']
    filename = 'RBFLN/scripts/models/' + filename
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
