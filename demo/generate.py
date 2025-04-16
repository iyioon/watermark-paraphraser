import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mersenne import mersenne_rng
from device_utils import get_device

EMAIL_PARAPHRASE_PROMPT_TEMPLATE = """
I need you to paraphrase the following email while preserving its exact meaning.

Instructions:
1. Rewrite using different vocabulary and phrasing
2. Maintain EXACTLY the same:
   - Email structure (greeting, body paragraphs, closing, signature)
   - All factual information, dates, numbers, and names
   - Professional tone and formality level
   - All contact information and references

Important guidelines:
- Keep the same paragraph structure
- Preserve any confidentiality notices or disclaimers exactly
- Do not add, remove, or modify any substantive information
- Maintain approximately the same length
- Start with the same type of greeting and end with similar closing

Original email:
{text}

Paraphrased version:
"""


def generate_shift(model, prompt, vocab_size, n, key, max_length=1000, verbose=True):
    rng = mersenne_rng(key)
    xi = torch.tensor([rng.rand()
                      for _ in range(n * vocab_size)]).view(n, vocab_size)
    shift = torch.randint(n, (1,))

    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    # Get the tokenizer to determine the EOS token ID
    tokenizer = model.config.tokenizer
    eos_token_id = model.config.eos_token_id

    # Store only the initial input length to exclude prompt from output
    prompt_length = inputs.size(1)

    # Initialize the output text
    previous_text = ""

    # Add separator line before generation starts
    if verbose:
        print("\n" + "=" * 80)
        print("GENERATING TEXT:")
        print("-" * 80 + "\n")

    i = 0
    # Continue generating until EOS token or max_length is reached
    while i < max_length:
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(
            output.logits[:, -1, :vocab_size], dim=-1).cpu()
        token = exp_sampling(probs, xi[(shift + i) % n, :]).to(model.device)

        # Stop if we generated an EOS token
        if token.item() == eos_token_id:
            break

        inputs = torch.cat([inputs, token], dim=-1)

        # Print the newly generated token
        if verbose:
            current_text = tokenizer.decode(
                inputs[0, prompt_length:], skip_special_tokens=True)
            new_text = current_text[len(previous_text):]
            print(new_text, end="", flush=True)
            previous_text = current_text

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        i += 1

    if verbose:
        print("\n" + "-" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80 + "\n")

    # Return only the generated tokens (exclude the prompt)
    return inputs[:, prompt_length:].detach().cpu()


def exp_sampling(probs, u):
    return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)


def main(args):
    torch.manual_seed(args.seed)
    device = get_device(verbose=args.verbose)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # Add the tokenizer to the model config for easy access
    model.config.tokenizer = tokenizer

    # Read text from the input document file
    try:
        with open(args.document, 'r') as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Error: Document file '{args.document}' not found.")
        return
    except Exception as e:
        print(f"Error reading document file: {e}")
        return

    # Create a paraphrasing prompt
    prompt_text = EMAIL_PARAPHRASE_PROMPT_TEMPLATE.format(text=input_text)

    tokens = tokenizer.encode(
        prompt_text, return_tensors='pt', truncation=True, max_length=2048)

    # Generate tokens with watermarking
    generated_tokens = generate_shift(
        model, tokens, len(tokenizer), args.n, args.key, verbose=args.verbose)[0]

    # Decode the generated tokens
    generated_text = tokenizer.decode(
        generated_tokens, skip_special_tokens=True)

    # If output file is specified, save only the generated text
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(generated_text)
            print(f"\nGenerated text saved to '{args.output}'")
        except Exception as e:
            print(f"\nError writing to output file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate text watermarked with a key')
    parser.add_argument('document', type=str,
                        help='a file containing text to paraphrase')
    parser.add_argument('--model', default='microsoft/phi-2', type=str,
                        help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--n', default=256, type=int,
                        help='the length of the watermark sequence')
    parser.add_argument('--key', default=42, type=int,
                        help='a key for generating the random watermark sequence')
    parser.add_argument('--seed', default=0, type=int,
                        help='a seed for reproducibile randomness')
    parser.add_argument('--output', type=str,
                        help='file to save the watermarked text to (optional)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print generation word-by-word')

    main(parser.parse_args())
