# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.


## What is inside

My implementation of the transformer model, including the encoder and decoder components, as well as the attention mechanism.

Also BPE trainer and tokenizer.

cs336_basics folder contains a lot of code, including the implementation of the transformer model and its components.

In order to run the code, you will need to set up your environment and install the necessary dependencies.

### Usage examples

Train BPE Tokenizer
```sh
python parallel_optim.py
```
Outputs:

../artifacts/.../vocab.json
../artifacts/.../merges.json

Encode Datasets to .npy Files
```sh
python bpe_main.py
```

Outputs:

../artifacts/.../train_tokens.npy
../artifacts/.../val_tokens.npy

Train

```sh
python cs336_basics/transformers_train.py \
  --train_data ./data/train_tokens.npy \
  --val_data ./data/val_tokens.npy \
  --vocab_size 50257 \
  --batch_size 32 \
  --context_length 1024 \
  --max_steps 100000
```

resume training

```sh
python cs336_basics/transformers_train.py \
  --train_data ./data/train_tokens.npy \
  --val_data ./data/val_tokens.npy \
  --vocab_size 50257 \
  --resume_from ./checkpoints/checkpoint_050000.pt \
  --checkpoint_dir ./checkpoints
```

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

```sh
uv run pytest
```



Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

