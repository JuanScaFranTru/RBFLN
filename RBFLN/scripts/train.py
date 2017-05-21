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
from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__)
