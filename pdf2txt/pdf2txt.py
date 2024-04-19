from pypdf import PdfReader
import re
import os


def main_local():
    with open("output.txt", "a", encoding='utf-8') as output:
        directory = 'documents'
        total_size = 0
        for document in os.listdir(directory):
            text_out, size = '', 0
            pdf = PdfReader(directory + '\\' + document)

            # get title from file name
            title = re.sub(r"-", " ", document[13:])
            title = re.sub(r"Paper(.*)pdf", "", title)
            size += output.write(title + '\t')

            # write each page individually to avoid buffer limit
            for page in pdf.pages:
                text = page.extract_text()
                text_out = re.sub(r"[\n\t\r]", " ", text)
                text_out = re.sub(r"\b\S{20,}\b", "", text_out)
                text_out = re.sub(r"\s+", " ", text_out)
                data = re.search(r"References \[1]", text_out)

                # remove the reference section of the paper
                if data is not None:
                    text_out, _ = re.split(r"References \[1]", text_out, 1)
                    size += output.write(text_out + ' ')
                    break
                size += output.write(text_out + ' ')
            output.write('\n')
            print(f'Size of document/line: {size}')
            total_size += size
        print(f'Size of file: {total_size}')


def examine_page():
    # update file name of PDF you wish to examine
    pdf = PdfReader('documents\\NeurIPS-2023-paintseg-painting-pixels-for-training-free-segmentation-Paper-Conference.pdf')
    page_num = 4
    page = pdf.pages[page_num]
    text = page.extract_text()
    text_out = re.sub(r"[\n\r]", " ", text)
    print(f'Text from page {page_num}:\n{text_out}')


def test():
    example = 'benfo 123nld9 q1US/iKKN gnd air'
    output = re.sub(r"\b\S{4,}\b", " ", example)
    print(f'Initial string: {example}\n'
          f'Cleaned string: {output}')


if __name__ == '__main__':
    # debug to check how certain pages are interpreted
    # examine_page()
    # debug to test regular expressions
    # test()
    # use local PDFs
    main_local()
