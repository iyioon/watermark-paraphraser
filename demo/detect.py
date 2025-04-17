import pyximport
import torch
from tqdm import tqdm
from mersenne import mersenne_rng
from transformers import AutoTokenizer
import numpy as np
import time
import argparse
import sys
import os
import concurrent.futures
from functools import partial
# Suppress the HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs': np.get_include()})


from levenshtein import levenshtein


def run_single_permutation(tokens, n, k, test_result, _):
    """Run a single permutation test for parallel processing"""
    vocab_size = tokens.shape[0]  # Get vocab size from tokens
    xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
    null_result = detect(tokens, n, k, xi_alternative)

    # Return 1 if the null result is less than or equal to the test result, 0 otherwise
    return null_result <= test_result


def permutation_test(tokens, key, n, k, vocab_size, n_runs=100, max_workers=None):
    """
    Run permutation test with parallel processing

    Args:
        tokens: Input token sequence to test
        key: Seed for the watermark sequence
        n: Length of watermark sequence
        k: Length of token sequence
        vocab_size: Size of vocabulary
        n_runs: Number of permutation runs
        max_workers: Maximum number of worker processes (None uses CPU count)
    """
    tokens = np.array(tokens, dtype=np.int32)

    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n * vocab_size)],
                  dtype=np.float32).reshape(n, vocab_size)
    test_result = detect(tokens, n, k, xi)

    # Use ProcessPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed parameters
        worker_fn = partial(run_single_permutation, tokens, n, k, test_result)

        # Map the worker function over the range of runs
        results = list(tqdm(
            executor.map(worker_fn, range(n_runs)),
            total=n_runs,
            desc="Running permutation tests"
        ))

    # Sum the results (True values count as 1)
    p_val = sum(results)

    return (p_val + 1.0) / (n_runs + 1.0)


def detect(tokens, n, k, xi, gamma=0.0):
    m = len(tokens)
    n = len(xi)

    A = np.empty((m - (k - 1), n))
    # Progress bar
    for i in tqdm(range(m - (k - 1)), desc="Computing detection matrix", leave=False):
        for j in range(n):
            A[i][j] = levenshtein(
                tokens[i:i + k], xi[(j + np.arange(k)) % n], gamma)

    return np.min(A)


def main(args):
    with open(args.document, 'r') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = tokenizer.encode(
        text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

    t0 = time.time()
    pval = permutation_test(tokens, args.key, args.n,
                            len(tokens), len(tokenizer),
                            n_runs=args.n_runs,
                            max_workers=args.workers)

    print('p-value: ', pval)
    print(f'(elapsed time: {time.time()-t0}s)')


if __name__ == '__main__':
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
    parser.add_argument('--n_runs', default=100, type=int,
                        help='number of permutation runs')
    parser.add_argument('--workers', default=None, type=int,
                        help='maximum number of worker processes (default: CPU count)')

    main(parser.parse_args())
