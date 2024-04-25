import pickle

with open("wiki_dpr/psgs_w100.tsv.pkl", 'rb') as file:
    data = file.readline()
    print(data)