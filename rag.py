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
    RagSequenceForGeneration,
    RagTokenizer,
)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split text into n-word passages"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
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


def main(
    rag_example_args: "RagExampleArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments"
):
    ######################################
    print("\nStep 1 - Create the dataset")
    ######################################

    assert os.path.isfile(rag_example_args.csv_path), "Please provide a valid path to a csv file"
    dataset = load_dataset(
        "csv", data_files=[rag_example_args.csv_path], split="train", delimiter="\t", column_names=["title", "text"]
    )

    # Split the documents into passages of 100 words
    dataset = dataset.map(split_documents, batched=True, num_proc=processing_args.num_proc)

    # Compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,
    )

    # And finally save your dataset
    passages_path = os.path.join(rag_example_args.output_dir, "dataset")
    dataset.save_to_disk(passages_path)

    ######################################
    print("\nStep 2 - Index the dataset")
    ######################################

    # Let's use the Faiss implementation of HNSW for  approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(rag_example_args.output_dir, "dataset_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)

    ######################################
    print("\nStep 3 - Load RAG")
    ######################################

    # Easy way to load the model
    # retriever = RagRetriever.from_pretrained(rag_example_args.rag_model_name, index_name="custom", indexed_dataset=dataset)
    retriever = RagRetriever.from_pretrained(rag_example_args.rag_model_name, index_name="custom", passages_path=passages_path, index_path=index_path)
    model = RagSequenceForGeneration.from_pretrained(rag_example_args.rag_model_name, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)

    ######################################
    print("\nStep 4 - Ask questions")
    ######################################

    if (rag_example_args.questions_file):
        questions = get_questions(rag_example_args.questions_file)
        for question in questions:
            print("\nQ: " + question)
            input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
            generated = model.generate(input_ids, max_length=rag_example_args.max_length, num_beams=rag_example_args.num_beams)
            generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            print("A: " + generated_string)
    else:
        while True:
            print("\nAsk a question:")
            question = input("Q: ")
            input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
            generated = model.generate(input_ids, max_length=rag_example_args.max_length, num_beams=rag_example_args.num_beams)
            generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            print("A: " + generated_string)


@dataclass
class RagExampleArguments:
    csv_path: str = field(
        default="datasets/rag_article.csv",
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    questions_file: Optional[str] = field(
        default=None,
        metadata={"help": "Questions that are passed as input to RAG."},
    )
    rag_model_name: str = field(
        default="facebook/rag-token-base",
        metadata={"help": "The RAG model to use."},
    )
    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={
            "help": (
                "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
                " 'facebook/dpr-ctx_encoder-multiset-base'"
            )
        },
    )
    output_dir: Optional[str] = field(
        default="processed_dataset",
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )
    max_length: Optional[int] = field(
        default=100,
        metadata={"help": "Max length for generator"},
    )
    num_beams: Optional[int] = field(
        default=2,
        metadata={"help": "Number of beams"},
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
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir
        main(rag_example_args, processing_args, index_hnsw_args)
