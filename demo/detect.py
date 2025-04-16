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
# Suppress the HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs': np.get_include()})


from levenshtein import levenshtein


def permutation_test(tokens, key, n, k, vocab_size, n_runs=100):

    tokens = np.array(tokens, dtype=np.int32)

    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n * vocab_size)],
                  dtype=np.float32).reshape(n, vocab_size)
    test_result = detect(tokens, n, k, xi)

    p_val = 0
    # Progress bar
    for run in tqdm(range(n_runs), desc="Running permutation tests", total=n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens, n, k, xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

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


def fast_permutation_test(tokens, key, n, k, vocab_size, n_runs=99):
    # Convert tokens to list
    tokens_list = tokens.tolist() if isinstance(
        tokens, np.ndarray) else list(tokens)

    # Create RNG with the given key
    rng = mersenne_rng(key)

    # Generate uniform random values
    u = np.array([rng.rand() for _ in range(n * vocab_size)], dtype=np.float32)

    # Calculate test statistic with actual watermark sequence
    test_result = fast_test_stat(tokens_list, u, n, k)

    # Map tokens to indices for faster processing
    unique_tokens = list(set(tokens_list))
    mapped_tokens = [unique_tokens.index(token) for token in tokens_list]

    p_val = 0
    for run in tqdm(range(n_runs), desc="Running fast permutation tests", total=n_runs):
        # Generate alternative random values
        u_alternative = np.random.random(
            n * len(unique_tokens)).astype(np.float32)

        # Calculate test statistic with alternative random values
        null_result = fast_test_stat(mapped_tokens, u_alternative, n, k)

        # Count how many null results are <= test result
        p_val += null_result <= test_result

    return (p_val + 1.0) / (n_runs + 1.0)


def fast_test_stat(tokens, u, n, k):
    vocab = len(u) // n
    m = len(tokens)

    A = []
    for i in tqdm(range(m - (k - 1)), desc="Computing fast detection matrix", leave=False):
        row = []
        for j in range(n):
            sub = [u[(vocab * j + p) % (vocab * n)] for p in range(vocab * k)]
            row.append(fast_levenshtein(tokens[i:i + k], sub, vocab))
        A.append(row)

    # Get minimum cost for each alignment
    closest = [min(row) for row in A]

    # Calculate median of all possible alignments
    closest.sort()
    mid = len(closest) // 2
    if len(closest) % 2 != 0:
        return closest[mid]
    else:
        return (closest[mid - 1] + closest[mid]) / 2


def fast_levenshtein(x, y, vocab, gamma=0.0):
    n = len(x)
    m = len(y) // vocab

    # Initialize matrix
    A = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Fill first row and column
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                A[i][j] = j * gamma
            elif j == 0:
                A[i][j] = i * gamma
            else:
                cost = np.log(1 - y[vocab * (j - 1) + x[i - 1]])
                A[i][j] = min(A[i - 1][j] + gamma, A[i][j - 1] +
                              gamma, A[i - 1][j - 1] + cost)

    return A[n][m]


def main(args):
    with open(args.document, 'r') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = tokenizer.encode(
        text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

    t0 = time.time()

    if args.fast:
        # Default k=10 for fast mode
        k_value = args.k if hasattr(args, 'k') else 10
        pval = fast_permutation_test(tokens, args.key, args.n,
                                     k_value, len(tokenizer), n_runs=args.n_runs)
    else:
        pval = permutation_test(tokens, args.key, args.n,
                                len(tokens), len(tokenizer), n_runs=args.n_runs)

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
    parser.add_argument('--fast', action='store_true',
                        help='use faster algorithm for detection')
    parser.add_argument('--k', default=10, type=int,
                        help='sequence length for levenshtein distance (only used with --fast)')
    parser.add_argument('--n_runs', default=100, type=int,
                        help='number of permutation test runs')

    main(parser.parse_args())
