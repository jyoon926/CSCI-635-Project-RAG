"""
A script creating a RAG checkpoint from a generator and a question encoder checkpoints.
"""

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, RagConfig, RagSequenceForGeneration, RagTokenForGeneration


def consolidate(
    model_type,
    generator: str,
    question_encoder: str,
    dest_dir: Path,
    config: str = None,
    generator_tokenizer: str = None,
    question_encoder_tokenizer: str = None,
):
    if config is None:
        config = "facebook/rag-token-base" if model_type == "rag_token" else "facebook/rag-sequence-base"

    if generator_tokenizer is None:
        generator_tokenizer = generator

    if question_encoder_tokenizer is None:
        question_encoder_tokenizer = question_encoder

    model_class = RagTokenForGeneration if model_type == "rag_token" else RagSequenceForGeneration

    # Save model.
    rag_config = RagConfig.from_pretrained(config)
    gen_config = AutoConfig.from_pretrained(generator)
    question_encoder_config = AutoConfig.from_pretrained(question_encoder)

    rag_config.generator = gen_config
    rag_config.question_encoder = question_encoder_config

    rag_model = model_class.from_pretrained_question_encoder_generator(
        question_encoder, generator, config=rag_config
    )
    rag_model.save_pretrained(dest_dir)

    # Sanity check.
    model_class.from_pretrained(dest_dir)

    # Save tokenizers.
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer)
    gen_tokenizer.save_pretrained(dest_dir / "generator_tokenizer/")
    question_encoder_tokenizer = AutoTokenizer.from_pretrained(question_encoder_tokenizer)
    question_encoder_tokenizer.save_pretrained(dest_dir / "question_encoder_tokenizer/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token"],
        required=True,
        type=str,
        help="RAG model type: rag_sequence, rag_token",
    )
    parser.add_argument("--dest", type=str, required=True, help="Path to the output checkpoint directory.")
    parser.add_argument("--generator", type=str, required=True, help="Generator model identifier")
    parser.add_argument(
        "--question_encoder", type=str, required=True, help="Question encoder model identifier"
    )

    parser.add_argument(
        "--generator_tokenizer",
        type=str,
        help="Generator tokenizer identifier, if not specified, resolves to ``generator``",
    )
    parser.add_argument(
        "--question_encoder_tokenizer",
        type=str,
        help="Question encoder tokenizer identifier, if not specified, resolves to ``question_encoder``",
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Identifier of the model config to use, if not provided, resolves to a base config for a given"
            " ``model_type``"
        ),
    )

    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(exist_ok=True)

    consolidate(
        args.model_type,
        args.generator,
        args.question_encoder,
        dest_dir,
        args.config,
        args.generator_tokenizer,
        args.question_encoder_tokenizer,
    )
