import sys
import codecs
import math


class WrongFileLength(Exception):
    pass


def file_len(data_path):
    with codecs.open(data_path, encoding='utf-8', mode='r') as fname:
        for i, l in enumerate(fname):
            pass
        print('File %s length: %s' % (data_path, str(i + 1)))
    return i + 1


def run(data_path1, data_path2, train_percent):
    input_file1 = codecs.open(data_path1, encoding='utf-8', mode='r')
    input_file2 = codecs.open(data_path2, encoding='utf-8', mode='r')
    output_file1 = codecs.open('train.inp', encoding='utf-8', mode='w')
    output_file2 = codecs.open('devel.inp', encoding='utf-8', mode='w')
    output_file3 = codecs.open('train.out', encoding='utf-8', mode='w')
    output_file4 = codecs.open('devel.out', encoding='utf-8', mode='w')

    a = file_len(data_path1)
    b = file_len(data_path2)
    if a != b:
        raise WrongFileLength('Input and output files must have the same length!')
    train_count = math.trunc(a * train_percent)
    print('Number of items for training %s' % str(train_count))

    sent_num = 0
    print('Splitting input file.')
    for line in input_file1:
        if sent_num < train_count:
            output_file1.write(line)
            sys.stdout.write("\rSentence number %s." % sent_num)
            sys.stdout.flush()
            sent_num += 1
        else:
            output_file2.write(line)
            sys.stdout.write("\rSentence number %s." % sent_num)
            sys.stdout.flush()
            sent_num += 1

    sent_num = 0
    print('Splitting output file.')
    for line in input_file2:
        if sent_num < train_count:
            output_file3.write(line)
            sys.stdout.write("\rSentence number %s." % sent_num)
            sys.stdout.flush()
            sent_num += 1
        else:
            output_file4.write(line)
            sys.stdout.write("\rSentence number %s." % sent_num)
            sys.stdout.flush()
            sent_num += 1

if __name__ == '__main__':
    path1 = str(sys.argv[1])
    path2 = str(sys.argv[2])
    train_percent = float(sys.argv[3])
    run(path1, path2, train_percent)