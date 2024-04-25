# CSCI-635-Project-RAG

This project implements the RAG framework and explores whether it is an effective approach to creating efficient large language models optimized for specific topics.

Run `pip install -r requirements.txt` to install all required packages.

Run the model with the default settings using:

`python3 rag.py`

# Recommended steps:

If you would like to initialize finetuning with a base model using different question encoder and generator architectures, you can build it with the consolidation script, e.g.:

`python3 consolidate_rag_checkpoint.py --model_type rag_token --generator facebook/bart-large-cnn --question_encoder facebook/dpr-question_encoder-multiset-base --dest ./rag_checkpoint`

You can then use it as the RAG model:

`python3 rag.py --rag_model ./rag_checkpoint`

You can also specify a custom dataset, as well as a questions file for bulk end-to-end generation.

`python3 rag.py --rag_model ./rag_checkpoint --questions_file datasets/questions.txt --dataset datasets/scrubbed_output_22APR.txt`

The original paper used the `wiki_dpr` dataset, which is a Wikipedia dump from 2018 consisting of 21M passages. Download the following files and place them in the `/wiki_dpr` folder.

<https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/psgs_w100.tsv.pkl>

<https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/hf_bert_base.hnswSQ8_correct_phi_128.c_index.index.dpr>

<https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/hf_bert_base.hnswSQ8_correct_phi_128.c_index.index_meta.dpr>

Then, run:

`python3 rag_wiki_dpr.py --rag_model ./rag_checkpoint --questions_file datasets/questions.txt`.