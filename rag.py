import os
from dataclasses import dataclass, field
from functools import partial
from tempfile import TemporaryDirectory
from typing import List, Optional
import faiss
import torch
from datasets import Features, Sequence, Value, load_dataset
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    HfArgumentParser,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    RagTokenizer
)
import time
import numpy as np

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n: str, character=" ") -> List[str]:
    """Split text into n-word passages"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text, hyper_params.passage_length):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizer) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def get_questions(questions_file: str):
    """Retrieve questions from file"""
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as file:
        for line in file:
            questions.append(line.strip())
    return questions


def get_output_filename():
    filename = "output/output"
    i = 0
    while os.path.exists(filename + str(i) + ".txt"):
        i += 1
    return filename + str(i) + ".txt"


def generate(tokenizer, model, question):
    input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    generated = model.generate(
                    input_ids, 
                    num_beams=hyper_params.num_beams,
                    max_length=hyper_params.max_length
                )
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return generated_string


def main(
    rag_args: "RagArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
    hyper_params: "HyperParameters"
):
    start_time = time.time()

    ######################################
    print("\nStep 1 - Create the dataset\n")
    ######################################

    assert os.path.isfile(rag_args.dataset), "Please provide a valid path to a csv file"
    dataset = load_dataset(
        "csv", data_files=[rag_args.dataset], split="train", delimiter="\t", column_names=["title", "text"]
    )

    # Split the documents into passages of 100 words
    dataset = dataset.map(split_documents, batched=True, num_proc=processing_args.num_proc)

    # Compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(rag_args.dpr_ctx_encoder).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(rag_args.dpr_ctx_encoder)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float64"))}
    )
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,
    )

    # And finally save your dataset
    passages_path = os.path.join(rag_args.dataset_output_dir, "dataset")
    dataset.save_to_disk(passages_path)

    ######################################
    print("\nStep 2 - Index the dataset\n")
    ######################################

    # Let's use the Faiss implementation of HNSW for approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(rag_args.dataset_output_dir, "dataset_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)

    indexing_time = time.time() - start_time
    print("-- Indexing the dataset took " + str(indexing_time) + " seconds --")

    ######################################
    print("\nStep 3 - Load RAG\n")
    ######################################
    
    load_start_time = time.time()

    # Easy way to load the model
    retriever = RagRetriever.from_pretrained(rag_args.rag_model, index_name="custom", indexed_dataset=dataset)
    retriever.n_docs = hyper_params.n_docs
    model = RagTokenForGeneration.from_pretrained(rag_args.rag_model, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_args.rag_model)

    loading_time = time.time() - load_start_time
    print("-- Loading the model took " + str(loading_time) + " seconds --")

    ######################################
    print("\nStep 4 - Ask questions")
    ######################################

    output_filename = get_output_filename()
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write("dataset: " + str(rag_args.dataset) + "\n")
        output_file.write("max_length: " + str(hyper_params.max_length) + "\n")
        output_file.write("num_beams: " + str(hyper_params.num_beams) + "\n")
        output_file.write("n_docs: " + str(hyper_params.n_docs) + "\n")
        output_file.write("passage_length: " + str(hyper_params.passage_length) + "\n")
        output_file.write("rag_model: " + str(rag_args.rag_model) + "\n")
        output_file.write("dpr_ctx_encoder: " + str(rag_args.dpr_ctx_encoder) + "\n\n")
        generation_times = []
        if (rag_args.questions_file):
            questions = get_questions(rag_args.questions_file)
            for question in questions:
                print("\nQ: " + question)
                q_time = time.time()
                generated_string = generate(tokenizer, model, question)
                elapsed = time.time() - q_time
                generation_times.append(elapsed)
                print("A: " + generated_string)
                print("-- Time to generate: " + str(elapsed) + " seconds --")
                output_file.write("Q: " + question + "\n\n")
                output_file.write("\tA: " + generated_string + "\n\n")
                output_file.write("\tTime to generate: " + str(elapsed) + " seconds\n\n")
        else:
            while True:
                print("\nAsk a question:")
                question = input("Q: ")
                q_time = time.time()
                generated_string = generate(tokenizer, model, question)
                elapsed = time.time() - q_time
                generation_times.append(elapsed)
                print("A: " + generated_string)
                print("-- Time to generate: " + str(elapsed) + " seconds --")
                output_file.write("Q: " + question + "\n\n")
                output_file.write("\tA: " + generated_string + "\n\n")
                output_file.write("\tTime to generate: " + str(elapsed) + " seconds\n\n")
        output_file.write("Total time: " + str(time.time() - start_time) + " seconds\n")
        output_file.write("Indexing time: " + str(indexing_time) + " seconds\n")
        output_file.write("Loading time: " + str(loading_time) + " seconds\n")
        output_file.write("Average generation time: " + str(np.average(generation_times)) + " seconds")


@dataclass
class RagArguments:
    dataset: str = field(
        default="datasets/rag_article.csv",
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    questions_file: Optional[str] = field(
        default=None,
        metadata={"help": "Questions that are passed as input to RAG."},
    )
    rag_model: str = field(
        default="facebook/rag-token-base",
        metadata={"help": "The RAG model to use."},
    )
    dpr_ctx_encoder: str = field(
        default="facebook/dpr-ctx_encoder-single-nq-base",
        metadata={
            "help": (
                "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
                " 'facebook/dpr-ctx_encoder-multiset-base'"
            )
        },
    )
    dataset_output_dir: str = field(
        default="processed_dataset",
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )
    
    
@dataclass
class HyperParameters:
    max_length: int = field(
        default=200,
        metadata={"help": "Max length for generator"},
    )
    num_beams: int = field(
        default=3,
        metadata={"help": "Number of beams"},
    )
    n_docs: int = field(
        default=10,
        metadata={"help": "Top k documents"},
    )
    passage_length: int = field(
        default=80,
        metadata={"help": "Length of passages in words"},
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={"help": "The dimension of the embeddings to pass to the HNSW Faiss index."},
    )
    m: int = field(
        default=256,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser((RagArguments, ProcessingArguments, IndexHnswArguments, HyperParameters))
    rag_args, processing_args, index_hnsw_args, hyper_params = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_args.dataset_output_dir = rag_args.dataset_output_dir or tmp_dir
        main(rag_args, processing_args, index_hnsw_args, hyper_params)
