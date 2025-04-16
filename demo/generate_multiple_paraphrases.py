import os
import argparse
import json
import torch
from generate import EMAIL_PARAPHRASE_PROMPT_TEMPLATE
from device_utils import get_device
from transformers import AutoTokenizer, AutoModelForCausalLM


class Args:
    """Simple class to hold arguments for the generate script."""

    def __init__(self, document, model, n, key, seed, output, verbose):
        self.document = document
        self.model = model
        self.n = n
        self.key = key
        self.seed = seed
        self.output = output
        self.verbose = verbose


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate multiple paraphrased versions of a document with different watermark keys'
    )
    parser.add_argument('document', type=str,
                        help='Path to the file containing text to paraphrase')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the watermarked texts to')
    parser.add_argument('--model', default='microsoft/phi-2', type=str,
                        help='HuggingFace model ID for generation')
    parser.add_argument('--n', default=256, type=int,
                        help='Length of the watermark sequence')
    parser.add_argument('--num-versions', default=3, type=int,
                        help='Number of paraphrased versions to generate')
    parser.add_argument('--seed', default=42, type=int,
                        help='Base seed for reproducible randomness')
    parser.add_argument('--key-range-start', default=100, type=int,
                        help='Starting value for watermark keys')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print generation details')

    return parser.parse_args()


def generate_multiple_paraphrases(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up the base filename for outputs
    base_filename = os.path.basename(args.document).split('.')[0]

    # List to store metadata for JSON
    metadata = []

    print(
        f"Generating {args.num_versions} paraphrased versions of {args.document}...")

    # Load model and tokenizer once to avoid reloading for each generation
    device = get_device(verbose=args.verbose)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.config.tokenizer = tokenizer

    # Read the input document once
    try:
        with open(args.document, 'r') as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Error: Document file '{args.document}' not found.")
        return
    except Exception as e:
        print(f"Error reading document file: {e}")
        return

    # Generate paraphrases with different keys
    for i in range(args.num_versions):
        # Generate a key using the range start
        key = args.key_range_start + i

        # Set output filename
        output_filename = os.path.join(
            args.output_dir, f"{base_filename}_key{key}.txt")

        print(
            f"Generating paraphrase {i+1}/{args.num_versions} with key {key}...")

        # Prepare prompt
        prompt_text = EMAIL_PARAPHRASE_PROMPT_TEMPLATE.format(text=input_text)
        tokens = tokenizer.encode(
            prompt_text, return_tensors='pt', truncation=True, max_length=2048
        ).to(device)

        # Set random seed for this generation
        generation_seed = args.seed + i
        torch.manual_seed(generation_seed)

        # Generate
        with torch.no_grad():
            from generate import generate_shift
            generated_tokens = generate_shift(
                model, tokens, len(tokenizer), args.n, key, verbose=args.verbose
            )[0]

            # Decode the generated text
            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)

            # Save to file
            with open(output_filename, 'w') as f:
                f.write(generated_text)

            # Add metadata
            metadata.append({
                "version": i + 1,
                "key": key,
                "seed": generation_seed,
                "output_file": output_filename,
                "model": args.model,
                "n": args.n
            })

            print(f"Saved paraphrase to {output_filename}")

    # Save metadata to JSON
    metadata_filename = os.path.join(
        args.output_dir, f"{base_filename}_metadata.json")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_filename}")
    print("All paraphrases generated successfully.")


if __name__ == '__main__':
    args = parse_args()
    generate_multiple_paraphrases(args)
