# Multi-Key Paraphrasing Watermarking for Document Distribution

This repository provides an implementation of a distortion‑free watermarking scheme for paraphrasing a confidential text into multiple watermarked versions. Each version is generated using a different secret key so that if one version is leaked, the corresponding key can be used to trace the source of the leak. A typical use case is detecting the leakage point of a confidential email distributed within a company.

## Overview

Implementation of the methods are described in [Robust Distortion-free Watermarks for Language Models](https://arxiv.org/abs/2307.15593).

by [**Rohith Kuditipudi**](https://web.stanford.edu/~rohithk/), [**John Thickstun**](https://johnthickstun.com/), [**Tatsunori Hashimoto**](https://thashim.github.io/), and [**Percy Liang**](https://cs.stanford.edu/~pliang/).

This repository contained modified code that implements the watermarks in [jthickstun/watermark](https://github.com/jthickstun/watermark)

## Setup

We provide a list of package requirements in `requirements.txt`, from which you can create a conda environment by running

```
conda create --name <env> --file requirements.txt
```

We recommend installing version `4.30.1` or earlier of the `transformers` package, as otherwise the roundtrip translation experiments may break.
Also, be sure to set the environment variables `$HF_HOME` and `$TRANSFORMERS_CACHE` to a directory with sufficient disk space.

## Basic Usage

We provide standalone Python code for generating and detecting text with a watermark, using our recommended instantiation of the watermarking strategies discussed in the paper in `demo/generate.py` and `demo/detect.py`. We also provide the Javascript implementation of the detector `demo/detect.js` used for the [in-browser demo](https://crfm.stanford.edu/2023/07/30/watermarking.html).

To generate `m` tokens of text from a model (e.g., `facebook/opt-1.3b`) with watermark key `42`, run:

```
python demo/generate.py data/in/email-1.txt --model facebook/opt-iml-1.3b --key 42 --output data/out/email-1.txt --verbose
```

Checking for the watermark requires a watermark key (in this case, `42`) and the model tokenizer, but crucially it does not require access to the model itself. To test for a watermark in a given text document `output.txt`, run

```
python demo/detect.py data/out/email-1.txt --tokenizer facebook/opt-iml-1.3b --key 42
```

Alternatively, you can use the javascript detector implemented in `demo/detect.js`.

## Experiment

To generate multiple paraphrased version of the same text (using different keys), run:

```
python demo/generate_multiple_paraphrases.py data/in/email-1.txt --output-dir data/out/email_1_paraphrases --num-versions 3 --model facebook/opt-iml-1.3b --verbose
```

This will generate 3 paraphrased versions of the input text `email-1.txt` and save them in the directory `data/out/email_1_paraphrases`. The generated files will be named `email-1_100.txt`, `email-1_101.txt`, and `email-1_102.txt`. `email-1_metadata.txt` will be available in the same directory.

To test the watermark in the generated paraphrased versions, run:

```
python demo/determine_watermarked_key.py data/out/email_1_paraphrases/email-1_key100.txt --keys 100 101 102 --verbose
```

This should return the key `100` as the watermark key for the file `email-1_key100.txt`.

## Analysis

To batch test the folder `data/out/email_1_paraphrases` for the watermark keys, run:

```
python demo/analyse.py data/out/email_1_paraphrases
```
