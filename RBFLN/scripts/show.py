"""Visualize results

Usage:
  show.py -i <file>
  show.py -h | --help

Options:
  -i <file>     Results file.
  -h --help     Show this screen.
"""
from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__)
