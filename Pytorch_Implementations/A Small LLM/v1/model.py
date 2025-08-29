from functools import partial
import os
from torch import nn
import torch
from torch import optim
from torchinfo import summary
from torch.optim.lr_scheduler import ExponentialLR

from ..data_utils import TrainDataset, collate_fn
from torch.utils.data import DataLoader


class EmbeddingLayers(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, max_pos_embed=768):
        super().__init__()
        # Layer for embedding the token
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Layer for embedding the position of each token inside the sequence.
        # Sets the size of the context window
        self.pos_embed = nn.Embedding(max_pos_embed, embed_dim)

        # For transformers, common practice
        self.token_embed.weight.data.normal_(mean=0.0, std=0.02)
        self.pos_embed.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_Seq):
        """
        Args:
            input_Seq (torch.Tensor): Tensor of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Embedded tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_Seq.shape
        positions = torch.arange(0, seq_len, device=input_Seq.device).unsqueeze(
            0
        )  # Shape: [1, seq_len]
        token_embeddings = self.token_embed(input_Seq)
        position_embeddings = self.pos_embed(positions).expand(
            batch_size, -1, -1
        )  # [B, T, D]
        return token_embeddings + position_embeddings


class CausalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Project full embedding dimension to full embedding dimension
        # These will be split into multiple heads later
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, input_seq):
        batch_size, seq_len, embed_dim = input_seq.shape

        # Apply linear projections to full embedding dimension
        Q = self.W_q(input_seq)  # (B, T, E)
        K = self.W_k(input_seq)  # (B, T, E)
        V = self.W_v(input_seq)  # (B, T, E)

        # Split into multiple heads and transpose for attention computation
        # (B, T, E) -> (B, T, NH, HD) -> (B, NH, T, HD)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention with causal masking
        attention = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=0.10
        )

        # Reshape back to original embedding size
        # (B, NH, T, HD) -> (B, T, NH, HD) -> (B, T, E)
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return attention


# Define MLP/ Feed Forward
class MLP(nn.Module):
    def __init__(self, embed_dim, expand_lvl):
        super().__init__()
        self.l1 = nn.Linear(embed_dim, expand_lvl * embed_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.15)
        self.l2 = nn.Linear(expand_lvl * embed_dim, embed_dim)

    def forward(self, input_seq):
        x = self.l1(input_seq)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.l2(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim, expand_lvl, num_heads):
        super().__init__()
        self.attn = CausalAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.FFN = MLP(embed_dim=embed_dim, expand_lvl=expand_lvl)
        self.ln1 = nn.LayerNorm(embed_dim, bias=False)
        self.ln2 = nn.LayerNorm(embed_dim, bias=False)

    def forward(self, input_seq):
        x = input_seq + self.attn(self.ln1(input_seq))
        x = x + self.FFN(self.ln2(x))
        return x


class LLM(nn.Module):
    def __init__(self, embed_dim, expand_lvl, n_blocks, num_heads, vocab_size):
        super().__init__()
        layers = [
            EmbeddingLayers(vocab_size=vocab_size, embed_dim=embed_dim),
        ]
        for i in range(n_blocks):
            layers.append(Block(embed_dim, expand_lvl, num_heads))
        layers.append(nn.Linear(embed_dim, vocab_size))

        self.main_model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main_model(x)
        return x


class LLM_Model:
    def __init__(
        self,
        embed_dim,
        expand_lvl,
        n_blocks,
        vocab_size,
        num_heads,
        epochs,
        device,
        tokenizer,
        max_lr=0.0004,
        lr_decay_exp=0.9,
    ):
        self.embed_dim = embed_dim
        self.expand_lvl = expand_lvl
        self.num_heads = num_heads
        self.epochs = epochs
        self.learning_rate = max_lr
        self.device = device
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.n_blocks = n_blocks

        print("Creating and Compiling the Model...")
        self.llm = torch.compile(
            LLM(
                self.embed_dim,
                self.expand_lvl,
                self.n_blocks,
                self.num_heads,
                self.vocab_size,
            ).to(self.device)
        )
        print("Compilation Finished!")

        # Define optimizer and scheduler
        self.optimizer = optim.AdamW(self.llm.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay_exp)

        # Loss function
        # Here ignore -100 (used for my padding tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def train(
        self,
        dataset,
        validation_dataset,
        train_batch_size=24,
        val_batch_size=16,
        train_max_seq_len=300,
        validation_max_seq_len=300,
    ):
        print("Starting Training!\n")
        ds = TrainDataset(
            tokenizer=self.tokenizer,
            dataset=dataset,
            max_seq_len=train_max_seq_len,
            min_length=15,
        )

        val_ds = TrainDataset(
            tokenizer=self.tokenizer,
            dataset=validation_dataset,
            max_seq_len=validation_max_seq_len,
            min_length=15,
        )

        train_loader = DataLoader(
            ds,
            batch_size=train_batch_size,
            collate_fn=partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id),
            num_workers=1,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            collate_fn=partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id),
            num_workers=1,
        )

        save_dir = "trained models/v1"
        os.makedirs(save_dir, exist_ok=True)

        start_epoch = self.load_checkpoint(save_dir)

        for epoch in range(start_epoch, self.epochs + 1):
            for batch_idx, batch_ex in enumerate(train_loader):
                data = batch_ex["input_ids"].to(self.device)
                labels = batch_ex["labels"].to(self.device)

                self.optimizer.zero_grad()
                logits = self.llm(data)
                loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch: {epoch} batch: {batch_idx + 1} train loss: {loss.item():.4f}"
                    )
                if batch_idx % 1000 == 0:
                    self.save_model(save_dir=save_dir, epoch=epoch)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.llm.parameters(), max_norm=1.0)

                # Perform a step
                self.optimizer.step()

            print(f"End of Epoch {epoch}\nRunning Validation Tests!\n")
            self.save_model(save_dir=save_dir, epoch=epoch)

            # After the inner training loop
            self.llm.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for val_batch in val_loader:
                    data_val = val_batch["input_ids"].to(self.device)
                    labels_val = val_batch["labels"].to(self.device)
                    logits_val = self.llm(data_val)
                    loss_val = self.loss_fn(
                        logits_val.view(-1, self.vocab_size),
                        labels_val.view(-1),
                    )
                    val_loss += loss_val.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}\n")
            self.llm.train()  # Restore to training mode

            self.scheduler.step()

    def model_summary(self):
        summary(self.llm, input_size=(1, 512), dtypes=[torch.int32])

    def load_checkpoint(self, save_dir):
        checkpoint_files = [
            f
            for f in os.listdir(save_dir)
            if f.startswith("model_epoch_") and f.endswith(".pt")
        ]

        start_epoch = 1
        if checkpoint_files:
            # Find the latest epoch
            latest_checkpoint = max(
                checkpoint_files, key=lambda x: int(x.split("_")[2].split(".")[0])
            )
            checkpoint_path = os.path.join(save_dir, latest_checkpoint)
            print(f"Loading checkpoint from {checkpoint_path}\n")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.llm.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # TODO: Temp. solution, enable the line below afterwards
            start_epoch = checkpoint["completed_epochs"]
            # start_epoch = checkpoint["completed_epochs"] + 1
            return start_epoch
        else:
            return 1

    def save_model(self, save_dir, epoch):
        # Save model after each epoch
        save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            {
                "completed_epochs": epoch,
                "model_state_dict": self.llm.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            save_path,
        )
        print(f"Saved model. Path: {save_path}")

    def generate_text(self, prompt, max_len=50):
        # Temporarily disable torch.compile for generation
        self.llm = self.llm._orig_mod if hasattr(self.llm, "_orig_mod") else self.llm

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        ).to(self.device)
        input_tokens = inputs["input_ids"]

        generated_ids = []
        for _ in range(max_len):
            with torch.no_grad():
                outputs = self.llm(input_tokens)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated_ids.append(next_token_id)
                input_tokens = torch.cat([input_tokens, next_token_id], dim=1)

        generated_text = self.tokenizer.decode(
            torch.cat(generated_ids, dim=1)[0], skip_special_tokens=True
        )
        # Re-enable torch.compile if needed
        # self.llm = torch.compile(self.llm)  # Only if you want to recompile later

        return generated_text
