# CSCI-635-Project-RAG

This project implements the RAG framework and explores whether it is an effective approach to creating efficient large language models optimized for specific topics.

Run the model on the rag_article.csv dataset using:

`python3 rag.py`

If you would like to initialize finetuning with a base model using different question encoder and generator architectures, you can build it with a consolidation script, e.g.:

`python3 consolidate_rag_checkpoint.py --model_type rag_token --generator_name_or_path facebook/bart-large-mnli --question_encoder_name_or_path facebook/dpr-question_encoder-multiset-base --dest ./rag_checkpoint`

Then you can use it as the RAG model:

`python3 rag.py --rag_model ./rag_checkpoint`