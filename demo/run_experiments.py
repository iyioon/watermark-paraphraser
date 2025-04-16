#!/usr/bin/env python3

import os
import subprocess
import argparse
import time
from pathlib import Path


def run_experiment(email_num, model="microsoft/phi-2", num_versions=3, verbose=True, starting_key=None):
    """
    Run paraphrase generation and analysis for a single email.

    Args:
        email_num (int): The email number (1-8)
        model (str): The model to use for paraphrasing
        num_versions (int): Number of paraphrase versions to generate
        verbose (bool): Whether to show verbose output
        starting_key (int): Starting key value for this email's watermarks
    """
    input_file = f"data/in/email-{email_num}.txt"
    output_dir = f"data/out/email_{email_num}_paraphrases"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(
            f"⚠️ Input file {input_file} does not exist. Skipping email-{email_num}.")
        return False

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Generate paraphrases
    print(f"\n{'='*80}")
    print(f"📝 Generating paraphrases for email-{email_num}")
    print(f"{'='*80}")

    generate_cmd = [
        "python", "demo/generate_multiple_paraphrases.py",
        input_file,
        "--output-dir", output_dir,
        "--num-versions", str(num_versions),
        "--model", model
    ]

    # Add starting key if specified
    if starting_key is not None:
        # Use the key-range-start parameter instead of keys
        generate_cmd.extend(["--key-range-start", str(starting_key)])
        print(f"Using key range starting from: {starting_key}")
        # Show which keys will be used
        keys = [starting_key + i for i in range(num_versions)]
        print(f"Keys that will be used: {keys}")

    if verbose:
        generate_cmd.append("--verbose")

    try:
        subprocess.run(generate_cmd, check=True)
        print(f"✅ Successfully generated paraphrases for email-{email_num}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating paraphrases for email-{email_num}: {e}")
        return False

    # Step 2: Analyze the paraphrases
    print(f"\n{'-'*80}")
    print(f"🔍 Analyzing paraphrases for email-{email_num}")
    print(f"{'-'*80}")

    analyse_cmd = [
        "python", "demo/analyse.py",
        output_dir
    ]

    try:
        subprocess.run(analyse_cmd, check=True)
        print(f"✅ Successfully analyzed paraphrases for email-{email_num}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error analyzing paraphrases for email-{email_num}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run paraphrase generation and analysis for multiple emails.')
    parser.add_argument('--emails', type=int, nargs='+', default=list(range(1, 9)),
                        help='Email numbers to process (default: 1-8)')
    parser.add_argument('--model', type=str, default='microsoft/phi-2',
                        help='Model to use for paraphrasing')
    parser.add_argument('--num-versions', type=int, default=3,
                        help='Number of paraphrase versions to generate')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable verbose output')
    parser.add_argument('--base-key', type=int, default=100,
                        help='Base key value to start with (default: 100)')
    parser.add_argument('--key-interval', type=int, default=100,
                        help='Interval between starting keys for each email (default: 100)')

    args = parser.parse_args()
    verbose = not args.no_verbose

    start_time = time.time()
    success_count = 0

    # Create base directories if they don't exist
    Path("data/out").mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting experiments for emails: {args.emails}")
    print(f"📋 Model: {args.model}")
    print(f"📋 Number of versions: {args.num_versions}")
    print(
        f"📋 Key range: Starting from {args.base_key} with interval {args.key_interval}")
    print(f"📋 Verbose: {verbose}")

    for email_num in args.emails:
        print(f"\n\n{'#'*80}")
        print(f"📧 Processing email-{email_num}")
        print(f"{'#'*80}")

        # Calculate starting key for this email
        starting_key = args.base_key + (email_num - 1) * args.key_interval
        print(f"Starting key for email-{email_num}: {starting_key}")

        if run_experiment(email_num, args.model, args.num_versions, verbose, starting_key):
            success_count += 1

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n\n{'#'*80}")
    print(f"📊 Experiment Summary")
    print(f"{'#'*80}")
    print(f"Total emails processed: {len(args.emails)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(args.emails) - success_count}")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
