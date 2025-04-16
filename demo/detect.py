import pyximport
import torch
from tqdm import tqdm
import numpy as np
import time
import argparse
import sys
import os
import concurrent.futures
from torch.cuda.amp import autocast

# Suppress the HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs': np.get_include()})

# Import after pyximport setup
from levenshtein import levenshtein
from mersenne import mersenne_rng

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def single_permutation_test(tokens, n, k, xi, vocab_size):
    """Run a single permutation test with random xi"""
    xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
    return detect(tokens, n, k, xi_alternative)


def permutation_test(tokens, key, n, k, vocab_size, n_runs=100):
    # Convert tokens to int32 for better performance
    tokens = np.array(tokens, dtype=np.int32)

    # Generate the watermark matrix
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n * vocab_size)],
                  dtype=np.float32).reshape(n, vocab_size)

    # Calculate test statistic for the actual watermark
    test_result = detect(tokens, n, k, xi)

    # Use concurrent.futures for parallel permutation tests
    p_val = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        futures = [executor.submit(single_permutation_test, tokens, n, k, xi, vocab_size)
                   for _ in range(n_runs)]

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=n_runs, desc="Running permutation tests"):
            null_result = future.result()
            # Assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result

    return (p_val + 1.0) / (n_runs + 1.0)


def batch_levenshtein(tokens, token_chunk, xi_chunk, n, k, gamma=0.0):
    """Process a batch of Levenshtein calculations"""
    results = np.empty(len(token_chunk))
    for idx, i in enumerate(token_chunk):
        min_dist = float('inf')
        for j in range(n):
            dist = levenshtein(
                tokens[i:i + k], xi_chunk[(j + np.arange(k)) % n], gamma)
            if dist < min_dist:
                min_dist = dist
        results[idx] = min_dist
    return results


def detect(tokens, n, k, xi, gamma=0.0):
    m = len(tokens)
    batch_size = 64  # Adjust based on memory constraints

    # Create a tensor to store all minimum distances
    min_distances = np.full(m - (k - 1), float('inf'))

    # Create chunks for parallel processing
    chunks = [(i, min(i + batch_size, m - (k - 1)))
              for i in range(0, m - (k - 1), batch_size)]

    # Process all rows first, finding minimum for each row
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for start, end in chunks:
            indices = list(range(start, end))
            futures.append(executor.submit(batch_levenshtein,
                           tokens, indices, xi, n, k, gamma))

        # Collect results as they complete
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            start, end = chunks[idx]
            min_distances[start:end] = future.result()

    return np.min(min_distances)


def main(args):
    with open(args.document, 'r') as f:
        text = f.read()

    # Use context manager to automatically handle GPU memory
    with torch.cuda.device(0):
        # Initialize tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        # Encode text
        tokens = tokenizer.encode(
            text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

        # Clear GPU cache to free memory
        torch.cuda.empty_cache()

        t0 = time.time()
        pval = permutation_test(tokens, args.key, args.n,
                                len(tokens), len(tokenizer), n_runs=100)

        threshold = 0.01

        print('p-value: ', pval)
        print(f'(elapsed time: {time.time()-t0}s)')


if __name__ == '__main__':
    # Set Windows-specific multiprocessing method
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking

    parser = argparse.ArgumentParser(
        description='test for a watermark in a text document')
    parser.add_argument('document', type=str,
                        help='a file containing the document to test')
    parser.add_argument('--tokenizer', default='microsoft/phi-2', type=str,
                        help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--n', default=256, type=int,
                        help='the length of the watermark sequence')
    parser.add_argument('--key', default=42, type=int,
                        help='the seed for the watermark sequence')

    main(parser.parse_args())
