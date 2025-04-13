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
python demo/generate.py data/in/document.txt --model facebook/opt-iml-1.3b --key 42 > data/out/output.txt
```

Checking for the watermark requires a watermark key (in this case, `42`) and the model tokenizer, but crucially it does not require access to the model itself. To test for a watermark in a given text document `output.txt`, run

```
python demo/detect.py data/out/output.txt --tokenizer facebook/opt-iml-1.3b --key 42
```

Alternatively, you can use the javascript detector implemented in `demo/detect.js` which runs much faster (this is also the detector used for the [web demo](https://crfm.stanford.edu/2023/07/30/watermarking.html)).

## Reproducing experiments from the paper

The [experiments](experiments) directory contains code for reproducing the experimental results we report in the paper (in particular, Experiments 1-7 as described in Section 3).
We use `experiments/c4-experiment.py` to run the news-like C4 dataset experiments (Experiments 1-6) and `experiments/instruct-experiment.py` to run the Alpaca instruction evaluation dataset experiments (Experiment 7).
In particular,

```
python experiments/{c4,instruct}-experiment.py --save results.p
```

will save experiment settings and results as a Python dict.
See `experiments/analyze.py` for a minimal example of how to parse this dict.

We include shell scripts for reproducing all experiments in [experiments/scripts](experiments/scripts). You will need to specify certain experiment settings in order to run most of the scripts.
For example, to reproduce Figure 2a in the paper using the OPT-1.3B model, run

```
./experiments/scripts/experiment-1.sh <save directory path> facebook/opt-1.3b
```

To reproduce Figure 4b in the paper using the LLaMA-7B model with `m = 35`, run

```
./experiments/scripts/experiment-3.sh arxiv-results/experiment-3/llama huggyllama/llama-7b 35
```

And as a final example, to reproduce Figure 9a in the paper using the LLaMA-7B model and with
French as the roundtrip language, run

```
./experiments/scripts/experiment-6.sh arxiv-results/experiment-6/opt facebook/opt-1.3b french
```

---

```bib
@article{kuditipudi2023robust,
  title={Robust Distortion-free Watermarks for Language Models},
  author={Kuditipudi, Rohith and Thickstun, John and Hashimoto, Tatsunori and Liang, Percy},
  journal={arXiv preprint arXiv:2307.15593},
  year={2023}
}
```
