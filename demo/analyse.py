import os
import sys
import json
import argparse
import re
import concurrent
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Import the function to determine watermarked keys
from determine_watermarked_key import determine_watermarked_keys


def extract_key_from_filename(filename):
    """
    Extract the key from a filename like 'email-1_key100.txt' or 'email-1_100.txt'.
    Returns None if no key is found.
    """
    # Try to match patterns like 'key100' or '_100'
    key_pattern1 = re.search(r'key(\d+)', filename)
    key_pattern2 = re.search(r'_(\d+)\.txt$', filename)

    if key_pattern1:
        return int(key_pattern1.group(1))
    elif key_pattern2:
        return int(key_pattern2.group(1))
    return None


def analyze_folder(folder_path, keys=None, threshold=0.01, tokenizer_name="microsoft/phi-2", n=128, verbose=False):
    """
    Analyze all text files in a folder to determine if they're watermarked with any of the provided keys.
    Uses parallel processing for faster analysis.

    Args:
        folder_path (str): Path to the folder containing text files
        keys (list): List of keys to check. If None, tries to get keys from metadata or filenames
        threshold (float): P-value threshold for determining watermark presence
        tokenizer_name (str): The name of the tokenizer to use
        n (int): The length of the watermark sequence
        verbose (bool): Whether to print detailed progress

    Returns:
        dict: Dictionary containing analysis results and accuracy metrics
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"'{folder_path}' is not a valid directory.")

    # Find all text files in the folder
    text_files = [f for f in os.listdir(folder_path) if f.endswith(
        '.txt') and not f.endswith('metadata.txt') and not f.endswith('analysis.txt')]

    if not text_files:
        raise ValueError(f"No text files found in '{folder_path}'.")

    # Try to determine keys from metadata file if not provided
    if keys is None:
        metadata_file = next((f for f in os.listdir(
            folder_path) if f.endswith('metadata.txt')), None)
        if metadata_file:
            try:
                with open(os.path.join(folder_path, metadata_file), 'r') as f:
                    metadata = json.load(f)
                    keys = metadata.get('keys', [])
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # If still no keys, try to extract from filenames
        if not keys:
            potential_keys = [extract_key_from_filename(f) for f in text_files]
            keys = sorted(set(k for k in potential_keys if k is not None))

        if not keys:
            raise ValueError(
                "No keys provided and couldn't determine keys from metadata or filenames.")

    print(
        f"Analyzing {len(text_files)} files against {len(keys)} keys: {keys}")

    # Initialize results dictionary
    results = {
        'files_analyzed': len(text_files),
        'keys_tested': keys,
        'file_results': {},
        'correct_identifications': 0,
        'total_files': 0,
        'accuracy': 0.0
    }

    # Define the worker function to analyze a single file
    def analyze_single_file(filename):
        file_path = os.path.join(folder_path, filename)
        actual_key = extract_key_from_filename(filename)

        try:
            matching_keys = determine_watermarked_keys(
                file_path, keys, threshold, tokenizer_name, n, verbose=False
            )

            # Store results for this file
            p_values = {}
            for key in keys:
                # Find the p-value for this key if it's in the matching_keys list
                p_value = next((p for k, p in matching_keys if k == key), None)
                if p_value is None:
                    # Run the detection to get the p-value even for non-matching keys
                    from detect import permutation_test
                    from transformers import AutoTokenizer

                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    tokens = tokenizer.encode(
                        text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
                    p_value = permutation_test(tokens, int(
                        key), n, len(tokens), len(tokenizer))

                if verbose:
                    print(
                        f"File: {filename} | Key: {key} | P-value: {p_value:.6f}")
                p_values[key] = p_value

            # Determine which key has the lowest p-value
            detected_key = min(p_values.items(), key=lambda x: x[1])[
                0] if p_values else None

            file_result = {
                'filename': filename,
                'actual_key': actual_key,
                'detected_key': detected_key,
                'p_values': p_values,
                'correct': actual_key is not None and detected_key == actual_key
            }

            return filename, file_result, actual_key

        except Exception as e:
            error_msg = str(e)
            print(f"Error analyzing {filename}: {error_msg}")
            return filename, {'error': error_msg}, actual_key

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks and collect futures
        future_results = {executor.submit(analyze_single_file, filename): filename
                          for filename in text_files}

        # Process results as they complete with progress bar
        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(text_files), desc="Analyzing files")

        for future in concurrent.futures.as_completed(future_results):
            if progress_bar:
                progress_bar.update(1)

            filename, file_result, actual_key = future.result()
            results['file_results'][filename] = file_result

            # Count correct identifications if we know the actual key
            if actual_key is not None:
                results['total_files'] += 1
                if file_result.get('correct', False):
                    results['correct_identifications'] += 1

        if progress_bar:
            progress_bar.close()

    # Calculate overall accuracy
    if results['total_files'] > 0:
        results['accuracy'] = results['correct_identifications'] / \
            results['total_files']

    return results


def create_heatmap(results, output_folder, filename="heatmap.png"):
    """
    Create a heatmap visualization and save it to a file.

    Args:
        results (dict): Analysis results
        output_folder (str): Folder where to save the image
        filename (str): Name of the png file to save

    Returns:
        str: Path to the saved heatmap image or None if visualization could not be created
    """
    try:
        # Create DataFrame for heatmap
        data = []
        for filename_key, file_result in results['file_results'].items():
            if 'p_values' not in file_result:
                continue

            for key, p_value in file_result['p_values'].items():
                data.append({
                    'Filename': filename_key,
                    'Key': key,
                    'P-value': p_value,
                    'Actual Key': file_result.get('actual_key'),
                    'Is Actual Key': key == file_result.get('actual_key')
                })

        if not data:
            return None

        df = pd.DataFrame(data)

        # Use a unique figure identifier to avoid name conflicts
        plt.figure(figsize=(
            12, max(8, len(results['file_results']) * 0.5)), num="watermark_heatmap")

        # Create a pivot table for the heatmap
        pivot_data = df.pivot(
            index='Filename', columns='Key', values='P-value')

        # Create the heatmap
        ax = sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3g',
            cmap="YlGnBu_r",
            cbar_kws={'label': 'P-value'},
            linewidths=.5
        )

        plt.title('P-values for Each File and Key Combination')
        plt.tight_layout()

        # Save figure to file
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)

        # Explicitly close any existing figure with this name
        plt.savefig(output_path, format='png', dpi=150)
        plt.close("watermark_heatmap")  # Close by figure number

        print(f"Heatmap successfully saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error creating heatmap visualization: {e}")
        return None


def save_analysis_report(results, output_path):
    """
    Save the analysis results to a Markdown file with linked visualization.

    Args:
        results (dict): Analysis results
        output_path (str): Path to save the report
    """
    # Create an images directory next to the output file
    output_dir = os.path.dirname(output_path)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Create heatmap visualization and save to file
    heatmap_filename = "heatmap.png"
    heatmap_path = create_heatmap(results, images_dir, heatmap_filename)

    # Get relative path for markdown link
    if heatmap_path:
        rel_heatmap_path = os.path.join('images', heatmap_filename)

    with open(output_path, 'w') as f:
        f.write("# Watermark Detection Analysis Report\n\n")

        # Overall summary
        f.write("## Summary\n\n")
        f.write(f"- Files analyzed: {results['files_analyzed']}\n")
        f.write(f"- Keys tested: {results['keys_tested']}\n")

        if results['total_files'] > 0:
            f.write(
                f"- Correct identifications: {results['correct_identifications']} out of {results['total_files']}\n")
            f.write(f"- Accuracy: {results['accuracy'] * 100:.2f}%\n\n")

        # Visualization - linked from the markdown
        if heatmap_path:
            f.write("## Visualization\n\n")
            f.write("P-values heatmap for each file and key combination:\n\n")
            f.write(f"![P-values Heatmap]({rel_heatmap_path})\n\n")
            f.write(
                "*Lower p-values (darker colors) indicate stronger evidence of watermarking with that key.*\n\n")

        # Results for each file
        f.write("## Detailed Results\n\n")

        for filename, file_result in results['file_results'].items():
            f.write(f"### {filename}\n\n")

            if 'error' in file_result:
                f.write(f"Error during analysis: {file_result['error']}\n\n")
                continue

            # Actual and detected keys
            f.write(f"- Actual key: {file_result['actual_key']}\n")
            f.write(f"- Detected key: {file_result['detected_key']}\n")
            f.write(
                f"- Correct identification: {'Yes' if file_result.get('correct', False) else 'No'}\n\n")

            # P-values table
            f.write("#### P-values for each key\n\n")
            p_values_table = [[key, f"{p_value:.6f}"]
                              for key, p_value in file_result['p_values'].items()]
            f.write(tabulate(p_values_table, headers=[
                    "Key", "P-value"], tablefmt="pipe"))
            f.write("\n\n")

            # Highlight the key with the lowest p-value
            min_p_key = min(
                file_result['p_values'].items(), key=lambda x: x[1])[0]
            f.write(f"The key with the lowest p-value is: **{min_p_key}**\n\n")

            # Conclusion
            if min_p_key == file_result['actual_key']:
                f.write("✅ Correctly identified the watermark key.\n\n")
            else:
                f.write("❌ Failed to correctly identify the watermark key.\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze watermarked text files to determine the keys used for watermarking.')
    parser.add_argument('folder', type=str,
                        help='Path to the folder containing watermarked text files')
    parser.add_argument('--keys', type=int, nargs='+',
                        help='List of keys to check against the files (optional)')
    parser.add_argument('--tokenizer', default='microsoft/phi-2', type=str,
                        help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--n', default=128, type=int,
                        help='the length of the watermark sequence')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='P-value threshold for determining watermark presence (default: 0.01)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--output', type=str,
                        help='Path to save the analysis report (default: <folder>/watermark_analysis.md)')

    args = parser.parse_args()

    # Set default output path if not provided
    if not args.output:
        args.output = os.path.join(args.folder, 'watermark_analysis.md')

    try:
        # Analyze the folder
        results = analyze_folder(
            args.folder,
            args.keys,
            args.threshold,
            args.tokenizer,
            args.n,
            args.verbose
        )

        # Save analysis report
        save_analysis_report(results, args.output)

        print(f"Analysis complete. Results saved to {args.output}")
        print(
            f"Accuracy: {results['accuracy'] * 100:.2f}% ({results['correct_identifications']} correct out of {results['total_files']})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
