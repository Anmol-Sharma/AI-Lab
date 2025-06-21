from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import os


def __train_tokenizer(
    dataset,
    vocab_size=32000,
    min_frequency=2,
    special_tokens=None,
    batch_size=2500,
    output_dir="./tokenizer",
):
    """
    Train a BPE tokenizer on a Hugging Face dataset with parquet files.
    Args:
        dataset: Hugging Face Dataset (with Text)
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for tokens
        special_tokens: List of special tokens
        batch_size: Batch size for processing
        output_dir: Directory to save the tokenizer
    """
    if special_tokens is None:
        special_tokens = ["<unk>", "<bos>", "</eos>", "<pad>", "<mask>"]

    # Get the training split
    train_dataset = dataset  # ['train']
    print(f"Dataset loaded with {len(train_dataset)} rows")

    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set pre-tokenizer (splits on whitespace)
    # For now this will do, for code and other formats like xml, json we need byte level splits
    # combined with whitespaces
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize the main BPE trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Yield batches to process
    def batch_iterator():
        batch = []
        for i, example in enumerate(train_dataset):
            # Extract text from the 'text' column
            text = example["text"]
            if text and isinstance(text, str) and len(text.strip()) > 0:
                batch.append(text)

            # Yield batch when it reaches batch_size
            if len(batch) >= batch_size:
                yield batch
                batch = []
        # Yield remaining items
        if batch:
            yield batch

    print("Starting tokenizer training...")
    # Train the tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Add post-processor for proper formatting
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A </eos>",
        pair="<bos> $A </eos> $B:1 </eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("</eos>", tokenizer.token_to_id("</eos>")),
        ],
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the tokenizer
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    # Convert to HuggingFace tokenizer format
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="</eos>",
        mask_token="<mask>",
    )
    # Save HuggingFace tokenizer
    hf_tokenizer.save_pretrained(output_dir)
    print("Tokenizer training completed!")
    print(f"Tokenizer saved to: {output_dir}")

    return hf_tokenizer


def test_tokenizer(tokenizer, test_texts=None):
    """Test the trained tokenizer with sample texts."""

    if test_texts is None:
        test_texts = [
            "Hello, how are you today?",
            "Machine learning and natural language processing are fascinating fields.",
        ]

    print("\n" + "=" * 50)
    print("TOKENIZER TESTING")
    print("=" * 50)

    for text in test_texts:
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        print(f"\nOriginal: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {decoded}")
        print(f"Number of tokens: {len(tokens)}")


def __load_existing_tokenizer(output_dir):
    """
    Load an existing tokenizer if it exists.

    Args:
        output_dir (str): Directory where tokenizer should be saved/loaded from

    Returns:
        tokenizer or None: Returns the loaded tokenizer if exists, None otherwise
    """
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    hf_tokenizer_path = os.path.join(output_dir, "tokenizer_config.json")

    # Check if both tokenizer files exist
    if os.path.exists(tokenizer_json_path) and os.path.exists(hf_tokenizer_path):
        try:
            # Load the HuggingFace tokenizer
            hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
            print(f"Loaded existing tokenizer from {output_dir}")
            return hf_tokenizer
        except Exception as e:
            print(f"Failed to load existing tokenizer: {e}")
            print("Will train a new tokenizer instead.")
            return None
    else:
        print("No existing tokenizer found")
        return None


# Updated training initiation code
def get_or_train_tokenizer(dataset, vocab_size, output_dir="./tokenizer_output"):
    """
    Get existing tokenizer or train a new one if it doesn't exist.

    Args:
        dataset: Training dataset
        output_dir (str): Directory to save/load tokenizer

    Returns:
        tokenizer: Either loaded existing or newly trained tokenizer
    """

    # Try to load existing tokenizer first
    existing_tokenizer = __load_existing_tokenizer(output_dir)

    if existing_tokenizer is not None:
        return existing_tokenizer
    else:
        # Train new tokenizer if none exists
        print("Training new tokenizer...")
        return __train_tokenizer(
            dataset=dataset, vocab_size=vocab_size, output_dir=output_dir
        )
