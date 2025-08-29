import os
from datasets import load_dataset
from .model import LLM_Model
from tokenizer_utils import get_or_train_tokenizer
import torch

EMBED_DIM = 768
VOCAB_SIZE = 28000
NUM_HEADS = 12
BLOCKS = 14
LR = 0.0005
LR_DECAY_GAMMA = 0.9
TRAIN_EPOCHS = 4
TRAIN_BATCH_SIZE = 24
VALIDATION_BATCH_SIZE = TRAIN_BATCH_SIZE // 2

# Define this for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "1"

device = "cpu"
if torch.mps.is_available():
    device = torch.device("mps")
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

    # Load the dataset for train/ validation split
    train_dataset = load_dataset(
        "wikimedia/wikipedia", name="20231101.en", split="train[:85%]", num_proc=4
    )

    validation_dataset = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.en",
        split="train[85%:88%]",
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
    )

    # llm_model.model_summary()

    llm_model.train(
        dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VALIDATION_BATCH_SIZE,
        train_max_seq_len=256,
        validation_max_seq_len=256,
    )
