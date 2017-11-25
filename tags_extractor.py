import sys
from spacy.en import English


def extract():
    input_file = sys.stdin
    output_file = sys.stdout
    nlp = English()
    for line in input_file:
        doc = nlp(line.strip())
        result = []
        for token in doc:
            if token.pos_ == 'PUNCT':
                result.append(token.orth_)
            else:
                if token.ent_type_ == "":
                    result.append('_' + token.tag_)
                else:
                    result.append('_NAME')
        if (result[-1] in ['.', '?', '!']) and (result[-2] not in ['.', '?', '!', '"', "'"]):
            result_str = ' '.join(result)
            output_file.write(result_str + '\n')


if __name__ == '__main__':
    extract()
