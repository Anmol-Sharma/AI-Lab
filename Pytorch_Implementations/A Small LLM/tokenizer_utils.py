from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
import os
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import unicodedata
import re
from tokenizers.processors import TemplateProcessing


def __train_english_wikipedia_tokenizer(
    dataset,
    vocab_size=30000,
    min_frequency=2,
    special_tokens=None,
    batch_size=2400,
    output_dir="./wikipedia_tokenizer",
):
    if special_tokens is None:
        special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>", "<mask>"]

    # Use BPE with ByteLevel for maximum robustness
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),  # Important!
    )

    def clean_wikipedia_text(text):
        """Robust cleaning for Wikipedia text"""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return None

        # Normalize unicode (fix some encoding issues)
        text = unicodedata.normalize("NFKD", text)

        # Remove or fix common mojibake patterns
        # Common mojibake replacements
        replacements = {
            "Ã©": "é",
            "Ã¨": "è",
            "Ãª": "ê",
            "Ã«": "ë",
            "Ã¡": "á",
            "Ã¢": "â",
            "Ã£": "ã",
            "Ã¤": "ä",
            "Ã¥": "å",
            "Ã­": "í",
            "Ã®": "î",
            "Ã¯": "ï",
            "Ã³": "ó",
            "Ã´": "ô",
            "Ãµ": "õ",
            "Ã¶": "ö",
            "Ãº": "ú",
            "Ã»": "û",
            "Ã¼": "ü",
            "Ã±": "ñ",
            "Ã§": "ç",
            # Add more as needed
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # Remove excessive control characters
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Remove excessive whitespace
        text = " ".join(text.split())

        return text if len(text) >= 10 else None

    def batch_iterator():
        batch = []
        processed = 0

        for example in dataset:
            text = example.get("text", "")
            cleaned_text = clean_wikipedia_text(text)

            if cleaned_text:
                batch.append(cleaned_text)
                processed += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    print("Training tokenizer on English Wikipedia...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    print(f"Training completed. Final vocab size: {tokenizer.get_vocab_size()}")

    # Post-processor
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    return tokenizer


def test_tokenizer(tokenizer, test_texts=None):
    """Test the trained tokenizer with sample texts."""

    if test_texts is None:
        test_texts = [
            "Hello, how are you today?",
            "Machine learning and natural language processing are fascinating fields.",
            "The Dark Lord: He who must not be named",
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

    # Check if both tokenizer files exist
    if os.path.exists(tokenizer_json_path):
        try:
            # Load the HuggingFace tokenizer
            hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
            # Explicitly tell it which tokens are special
            hf_tokenizer.pad_token = "<pad>"
            hf_tokenizer.unk_token = "<unk>"
            hf_tokenizer.bos_token = "<bos>"
            hf_tokenizer.eos_token = "<eos>"
            hf_tokenizer.mask_token = "<mask>"

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
        return __train_english_wikipedia_tokenizer(
            dataset=dataset, vocab_size=vocab_size, output_dir=output_dir
        )
