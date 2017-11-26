import sys
import codecs
import re
import nltk
import random
import spacy
from nltk.corpus import wordnet as wn
from itertools import chain


def run(data_path):
    nlp = spacy.load('en')
    input_file = codecs.open(data_path, encoding='utf-8', mode='r')
    output_file1 = codecs.open('input.txt', encoding='utf-8', mode='w')
    output_file2 = codecs.open('output.txt', encoding='utf-8', mode='w')
    sent_num = 0
    for line in input_file:
        doc = nlp(line.strip())
        result1 = []
        result2 = []
        result3 = []
        result4 = []
        for token in doc:
            if token.ent_type_ != "":
                tag_ = '_NAME'
                result1.append(tag_)
                result2.append(tag_)
                result3.append(tag_)
                result4.append(tag_)
                line = line.replace(token.orth_, tag_)
            else:
                if token.pos_ == 'PUNCT':
                    result1.append(token.orth_)
                    result2.append(token.orth_)
                    result3.append(token.orth_)
                    result4.append(token.orth_)
                else:
                    if random.randrange(1, 10) <= 4:
                        result1.append(token.lemma_)
                    else:
                        result1.append('_' + token.tag_)

                    if random.randrange(1, 10) <= 5:
                        result2.append(token.lemma_)
                    else:
                        result2.append('_' + token.tag_)

                    if random.randrange(1, 10) <= 6:
                        result3.append(token.lemma_)
                    else:
                        result3.append('_' + token.tag_)

                    if token.pos_ in ['NOUN', 'VERB', 'PRON', 'ADJ']:
                        result4.append(token.lemma_)
                    else:
                        result4.append('_' + token.pos_)

        if len(result1) >= 2:
            result_str1 = ' '.join(result1)
            result_str1 = result_str1 + '\n'
            result_str2 = ' '.join(result2)
            result_str2 = result_str2 + '\n'
            result_str3 = ' '.join(result3)
            result_str3 = result_str3 + '\n'
            result_str4 = ' '.join(result4[:-1])
            result_str4 = result_str4 + ' ' + line[-2:]
            output_file1.write(result_str1)
            output_file1.write(result_str2)
            output_file1.write(result_str3)
            output_file1.write(result_str4)
            output_file2.write(line)
            output_file2.write(line)
            output_file2.write(line)
            output_file2.write(line)
            sent_num += 1
            sys.stdout.write("\rSentence number %s." % sent_num)
            sys.stdout.flush()
        else:
            continue


if __name__ == '__main__':
    path = str(sys.argv[1])
    run(path)
