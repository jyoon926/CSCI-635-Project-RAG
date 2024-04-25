import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
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
    hyper_params: "HyperParameters"
):
    start_time = time.time()
    
    ######################################
    print("\nStep 1 - Load RAG\n")
    ######################################

    # Easy way to load the model
    retriever = RagRetriever.from_pretrained(rag_args.rag_model, index_path=rag_args.index_path, index_name="custom")
    retriever.n_docs = hyper_params.n_docs
    model = RagTokenForGeneration.from_pretrained(rag_args.rag_model, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_args.rag_model)

    loading_time = time.time() - start_time
    print("-- Loading the model took " + str(loading_time) + " seconds --")

    ######################################
    print("\nStep 2 - Ask questions")
    ######################################

    output_filename = get_output_filename()
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write("dataset: wiki_dpr\n")
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
        output_file.write("Loading time: " + str(loading_time) + " seconds\n")
        output_file.write("Average generation time: " + str(np.average(generation_times)) + " seconds")


@dataclass
class RagArguments:
    index_path: str = field(
        default="./wiki_dpr",
        metadata={"help": "The path to the indexed dataset."},
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


if __name__ == "__main__":
    parser = HfArgumentParser((RagArguments, HyperParameters))
    rag_args, hyper_params = parser.parse_args_into_dataclasses()
    main(rag_args, hyper_params)
