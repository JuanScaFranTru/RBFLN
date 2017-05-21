"""Evaluate rbfln model using the test set.

Usage:
  eval.py -i <file> -o <file>
  eval.py -h | --help

Options:
  -i <file>     Model file.
  -o <file>     Results file.
  -h --help     Show this screen.
"""
from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__)
