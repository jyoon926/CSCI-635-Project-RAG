import os
import re


def main():
    with open("scrubbed_output_test.txt", "a", encoding='utf-8') as output:
        directory = 'copy_paste_text'
        for document in os.listdir(directory):
            with open(directory + '\\' + document, 'r', encoding='utf-8') as dirty_text:
                first = True
                for line in dirty_text:
                    line = line.strip()
                    if first:
                        output.write(line + '\t')
                        first = False
                    else:
                        line = re.sub(r"\[[ 0-9]*]", "", line)
                        line = re.sub(r"[\n\t\r]", " ", line)
                        line = re.sub(r"\s+", " ", line)
                        output.write(line + ' ')
                output.write('\n')


if __name__ == '__main__':
    main()
