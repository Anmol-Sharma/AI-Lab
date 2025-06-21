from torch.utils.data import Dataset
import torch
import random
from torch.nn.utils.rnn import pad_sequence


class TrainDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        max_seq_len: int,
        min_length=10,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_length = min_length
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text = self.dataset[idx].get("text", "")
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) < self.min_length:
            # Try a nearby item
            new_idx = (idx + 1) % len(self.dataset)
            return self.__getitem__(new_idx)

        # Sample a random window
        max_start = max(0, len(tokens) - self.min_length)
        start = random.randint(
            0, min(max_start, max(0, len(tokens) - self.max_seq_len))
        )
        end = min(start + self.max_seq_len, len(tokens))
        tokens_window = tokens[start:end]

        input_ids = torch.tensor(tokens_window, dtype=torch.long)

        return {
            "input_ids": input_ids,
        }


def collate_fn(batch, pad_token_id):
    """
    Helper function to combine together items in a batch with varied lengths
    """
    input_ids = [item["input_ids"] for item in batch]

    # Pad sequences to the max length in the batch
    padded_input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )

    # Create labels (same as input_ids for causal LM)
    labels = padded_input_ids.clone()

    # Shift left the labels to train model to predict the next token in the sequence
    labels[:, :-1] = padded_input_ids[:, 1:]
    # Step 3: Set the last token in labels to -100 (no corresponding input token)
    labels[:, -1] = -100  # CrossEntropyLoss will ignore this

    # Step 4: Replace padding tokens in labels with -100
    labels[labels == pad_token_id] = -100

    return {
        "input_ids": padded_input_ids,
        "labels": labels,
    }
