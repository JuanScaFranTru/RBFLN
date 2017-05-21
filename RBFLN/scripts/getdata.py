"""Process data from a csv file

Usage:
  getdata.py -i <file> -o <file>
  getdata.py -h | --help

Options:
  -i <file>     Input raw data file.
  -o <file>     Output data file.
  -h --help     Show this screen.
"""
from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__)
