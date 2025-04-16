import concurrent.futures
import argparse
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys

from detect import permutation_test


def get_detection_p_value(text, key, tokenizer_name="microsoft/phi-2", n=256):
    """
    Get the p-value for detecting a watermark in text using a specific key.

    Args:
        text (str): The text to check for watermarks
        key (int): The watermark key to check
        tokenizer_name (str): The name of the tokenizer to use
        n (int): The length of the watermark sequence

    Returns:
        float: The p-value indicating likelihood of watermark presence
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the text
    tokens = tokenizer.encode(
        text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

    # Run the permutation test
    p_value = permutation_test(tokens, int(
        key), n, len(tokens), len(tokenizer))

    return p_value


def determine_watermarked_keys(input_file, keys, threshold=0.01, tokenizer_name="microsoft/phi-2", n=256, verbose=False):
    """
    Determine which keys (if any) were used to watermark the input file.
    Processes keys in parallel to improve performance.

    Args:
        input_file (str): Path to the input file to check
        keys (list): List of keys to check against the input file
        threshold (float): P-value threshold for determining watermark presence
        tokenizer_name (str): The name of the tokenizer to use
        n (int): The length of the watermark sequence
        verbose (bool): Whether to print detailed results for each key

    Returns:
        list: List of tuples containing (key, p_value) for keys with p-value < threshold
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read the input file content
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the text once
    tokens = tokenizer.encode(
        text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

    vocab_size = len(tokenizer)

    # Define a worker function to process a single key
    def process_key(key):
        t0 = time.time()
        p_value = permutation_test(
            tokens, int(key), n, len(tokens), vocab_size)
        elapsed_time = time.time() - t0

        if verbose:
            print(
                f"Key: {key}, p-value: {p_value:.6f} (elapsed time: {elapsed_time:.2f}s)")

        return (key, p_value)

    matching_keys = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_key = {executor.submit(
            process_key, key): key for key in keys}

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_key),
            total=len(keys),
            desc="Testing keys",
            disable=not verbose
        ):
            try:
                key, p_value = future.result()
                if p_value < threshold:
                    matching_keys.append((key, p_value))
            except Exception as e:
                if verbose:
                    key = future_to_key[future]
                    print(f"Error processing key {key}: {e}")

    return matching_keys


def main():
    parser = argparse.ArgumentParser(
        description='Determine if a file has been watermarked with any of the provided keys.')
    parser.add_argument('document', type=str,
                        help='Path to the input file to check')
    parser.add_argument('--keys', type=int, nargs='+', required=True,
                        help='List of keys to check against the input file')
    parser.add_argument('--tokenizer', default='microsoft/phi-2', type=str,
                        help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--n', default=256, type=int,
                        help='the length of the watermark sequence')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='P-value threshold for determining watermark presence (default: 0.01)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each key')

    args = parser.parse_args()

    try:
        print(
            f"Checking document '{args.document}' against {len(args.keys)} keys...")

        matching_keys = determine_watermarked_keys(
            args.document,
            args.keys,
            args.threshold,
            args.tokenizer,
            args.n,
            args.verbose
        )

        if matching_keys:
            print("\nWatermark detected with the following keys:")
            for key, p_value in matching_keys:
                print(f"  - Key: {key}, p-value: {p_value:.6f}")
        else:
            print("\nNo watermark detected with any of the provided keys.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
