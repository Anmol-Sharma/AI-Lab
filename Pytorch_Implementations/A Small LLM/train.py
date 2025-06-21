# Run using :- torchrun --nproc_per_node=2 train.py
# Modify the parameter depending on the number of GPUs available

import os
from datasets import load_dataset
from tokenizer_utils import get_or_train_tokenizer
from model import LLM_Model
import torch

EMBED_DIM = 768
VOCAB_SIZE = 32000
NUM_HEADS = 12
BLOCKS = 12
LR = 0.0003
LR_DECAY_GAMMA = 0.9
TRAIN_EPOCHS = 4


# Define this for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "1"

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
print("Device in use :", device)


def get_trained_tokenizer():
    # Load the complete dataset for training the tokenizer if necessary
    dataset = load_dataset(
        "wikimedia/wikipedia", name="20231101.en", split="train", num_proc=4
    )
    # Get tokenizer
    trained_tokenizer = get_or_train_tokenizer(dataset, VOCAB_SIZE)
    return trained_tokenizer


if __name__ == "__main__":
    tokenizer = get_trained_tokenizer()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"WORLD Size: {world_size}")

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Update the dataset for train/ validation split
    train_dataset = load_dataset(
        "wikimedia/wikipedia", name="20231101.en", split="train[:85%]", num_proc=4
    )
    validation_dataset = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.en",
        split="train[85%:90%]",
        num_proc=2,
    )

    llm_model = LLM_Model(
        embed_dim=EMBED_DIM,
        expand_lvl=2,
        n_blocks=BLOCKS,
        vocab_size=VOCAB_SIZE,
        num_heads=NUM_HEADS,
        epochs=TRAIN_EPOCHS,
        device=device,
        tokenizer=tokenizer,
        max_lr=LR,
        lr_decay_exp=LR_DECAY_GAMMA,
        world_size=world_size,
        rank=rank,
    )

    llm_model.train(
        dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_batch_size=32,
        val_batch_size=8,
    )
