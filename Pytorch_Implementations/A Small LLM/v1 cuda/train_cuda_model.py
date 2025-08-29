# pip3 install tokenizers datasets transformers torchinfo
# Run using :- torchrun --nproc_per_node=<num-gpus> train_cuda.py
# Modify the parameter depending on the number of GPUs available

import os
from datasets import load_dataset
from model_cuda import LLM_Model
from tokenizer_utils import get_or_train_tokenizer
import torch

EMBED_DIM = 768
VOCAB_SIZE = 28000
NUM_HEADS = 12
BLOCKS = 18
LR = 0.0004
LR_DECAY_GAMMA = 0.9
TRAIN_EPOCHS = 7
TRAIN_BATCH_SIZE = 48
VALIDATION_BATCH_SIZE = TRAIN_BATCH_SIZE // 2

# Define this for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "1"


def get_trained_tokenizer():
    # Load the complete dataset for training the tokenizer if necessary
    dataset = load_dataset(
        "wikimedia/wikipedia", name="20231101.en", split="train", num_proc=4
    )
    # Get tokenizer
    trained_tokenizer = get_or_train_tokenizer(dataset, VOCAB_SIZE)
    return trained_tokenizer


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert torch.cuda.is_available(), "CUDA is required"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Setup downloading dataset if not already loaded
    if rank == 0:
        load_dataset(
            "wikimedia/wikipedia", name="20231101.en", split="train", num_proc=4
        )
    torch.distributed.barrier()

    # Explicitly set the device for this process ---
    torch.cuda.set_device(rank)

    tokenizer = get_trained_tokenizer()
    # Update the dataset for train/ validation split
    train_dataset = load_dataset(
        "wikimedia/wikipedia", name="20231101.en", split="train[:85%]", num_proc=4
    )

    validation_dataset = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.en",
        split="train[85%:48%]",
        num_proc=2,
    )

    # --- Synchronize all processes ---
    # All other ranks will wait here until rank 0 has finished downloading the data.
    # They will then load the data from the cache.
    torch.distributed.barrier()

    llm_model = LLM_Model(
        embed_dim=EMBED_DIM,
        expand_lvl=2,
        n_blocks=BLOCKS,
        vocab_size=VOCAB_SIZE,
        num_heads=NUM_HEADS,
        epochs=TRAIN_EPOCHS,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        max_lr=LR,
        lr_decay_exp=LR_DECAY_GAMMA,
    )

    # llm_model.model_summary()

    llm_model.train(
        dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VALIDATION_BATCH_SIZE,
        train_max_seq_len=512,
        validation_max_seq_len=512,
    )
