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

    # Move data to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens_tensor = torch.tensor(tokens, device=device)
    u_tensor = torch.tensor(u, device=device, dtype=torch.float32)

    # Process in batches to maximize GPU utilization
    batch_size = 500  # Reduced to avoid memory issues
    num_batches = (m - (k - 1) + batch_size - 1) // batch_size
    closest_distances = []

    for batch in tqdm(range(num_batches), desc="Computing fast detection matrix"):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, m - (k - 1))

        batch_min_distances = torch.ones(
            end_idx - start_idx, device=device) * float('inf')

        for i in range(start_idx, end_idx):
            idx = i - start_idx
            # Get current token sequence
            seq = tokens_tensor[i:i + k]

            min_dist = float('inf')
            for j in range(n):
                # Extract pattern
                sub = torch.zeros(vocab * k, device=device,
                                  dtype=torch.float32)
                for p in range(vocab * k):
                    sub[p] = u_tensor[(vocab * j + p) % (vocab * n)]

                # Calculate distance using vectorized operations where possible
                distance = vectorized_levenshtein(seq, sub, vocab, device)
                min_dist = min(min_dist, distance)

            batch_min_distances[idx] = min_dist

        closest_distances.extend(batch_min_distances.cpu().numpy().tolist())
        # Clear GPU cache to prevent memory issues
        torch.cuda.empty_cache()

    # Calculate median
    closest_distances.sort()
    mid = len(closest_distances) // 2
    if len(closest_distances) % 2 != 0:
        return closest_distances[mid]
    else:
        return (closest_distances[mid - 1] + closest_distances[mid]) / 2.0


def vectorized_levenshtein(x, y, vocab, device, gamma=0.0):
    """
    Optimized Levenshtein distance calculation using GPU
    """
    n = len(x)
    m = len(y) // vocab

    # Initialize DP matrix on GPU - using a single allocation
    dp = torch.zeros((n + 1, m + 1), device=device, dtype=torch.float32)

    # Fill first row and column with vectorized operations
    dp[0, :] = torch.arange(0, m + 1, device=device) * gamma
    dp[:, 0] = torch.arange(0, n + 1, device=device) * gamma

    # More efficient implementation with fewer item() calls
    x_cpu = x.cpu().numpy()  # Copy to CPU once, not in the loop

    # Process the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            token_idx = x_cpu[i - 1]
            cost = torch.log(1 - y[vocab * (j - 1) + token_idx])

            deletion = dp[i - 1, j] + gamma
            insertion = dp[i, j - 1] + gamma
            substitution = dp[i - 1, j - 1] + cost

            dp[i, j] = torch.min(torch.min(deletion, insertion), substitution)

    return dp[n, m].item()


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
