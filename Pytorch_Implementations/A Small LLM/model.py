from functools import partial
import os
from torch import nn
import torch
from torch import optim
from torchinfo import summary
from torch.optim.lr_scheduler import ExponentialLR

from data_utils import TrainDataset, collate_fn
from torch.utils.data import DataLoader

from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class EmbeddingLayers(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, max_pos_embed=768):
        super().__init__()
        # Layer for embedding the token
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Layer for embedding the position of each token inside the sequence.
        # Sets the size of the context window
        self.pos_embed = nn.Parameter(torch.randn(max_pos_embed, embed_dim))

    def forward(self, input_Seq):
        """
        Args:
            input_Seq (torch.Tensor): Tensor of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Embedded tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_Seq.shape
        token_embeddings = self.token_embed(input_Seq)  # Direct lookup
        position_embeddings = self.pos_embed[:seq_len, :]  # Slice to seq_len

        return token_embeddings + position_embeddings


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Necessary to split the full embedding dimension into number of heads
        # Each head attends to part of the embedding dim
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, input_seq):
        batch_size, seq_len, embed_dim = input_seq.shape

        # Split the input into multiple heads
        input_seq = input_seq.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )  # (B, T, NH, HD)

        # Apply the first level projection
        Q = self.W_q(input_seq)  # B, T, NH, HD
        K = self.W_k(input_seq)  # B, T, NH, HD
        V = self.W_v(input_seq)  # B, T, NH, HD

        # Transpose to get proper shape for attention computation
        # (B, T, NH, HD) -> (B, NH, T, HD)
        Q = Q.transpose(1, 2)  # B, NH, T, HD
        K = K.transpose(1, 2)  # B, NH, T, HD
        V = V.transpose(1, 2)  # B, NH, T, HD

        # Compute Scaled dot product attention
        # Each head attention to set of tokens
        attention = torch.matmul(Q, K.transpose(-2, -1))  # B, NH, T, T

        # Create mask to prevent attending to future tokens i.e. for each token, only attend tokens
        # Prior which have been processed
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.W_q.weight.device)

        # Expand mask to suite dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # 1, 1, T, T

        # Replace 0 with -inf
        attention = attention.masked_fill(mask == 0, float("-inf"))

        # Update the token to token attention with this given mask
        attention = torch.softmax(
            attention / (self.head_dim**0.5), dim=-1
        )  # Apply softmax on last dim

        # Weighted sum
        attention = torch.matmul(
            attention, V
        )  # (B, NH, T, T) @ (B, NH, T, HD) -> (B, NH, T, HD)

        # Reshape back to original embedding size
        # (B, NH, T, HD) -> (B, T, NH * HD)
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
        self.attn = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
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
        world_size,
        rank,
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
        self.world_size = world_size
        self.rank = rank

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

        self.llm = DDP(self.llm, device_ids=[self.rank])

        # Define optimizer and scheduler
        self.optimizer = optim.AdamW(self.llm.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay_exp)

        # Loss function
        # Here ignore -100 (used for my padding tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.scaler = GradScaler()

    def train(
        self, dataset, validation_dataset, train_batch_size=24, val_batch_size=16
    ):
        print("Starting Training!")
        ds = TrainDataset(
            tokenizer=self.tokenizer, dataset=dataset, max_seq_len=768, min_length=10
        )

        val_ds = TrainDataset(
            tokenizer=self.tokenizer,
            dataset=validation_dataset,
            max_seq_len=768,
            min_length=8,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False
        )

        train_loader = DataLoader(
            ds,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id),
            num_workers=1,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            sampler=val_sampler,
            collate_fn=partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id),
            num_workers=1,
        )

        save_dir = "models"
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        torch.distributed.barrier()

        start_epoch = self.load_checkpoint(save_dir)

        for epoch in range(start_epoch, self.epochs + 1):
            train_sampler.set_epoch(epoch)
            for batch_idx, batch_ex in enumerate(train_loader):
                data = batch_ex["input_ids"].to(self.device)
                labels = batch_ex["labels"].to(self.device)

                self.optimizer.zero_grad()

                # Mixed precision context using bfloat16
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.llm(data)
                    loss = self.loss_fn(
                        logits.view(-1, self.vocab_size), labels.view(-1)
                    )

                if batch_idx % 2500 == 0:
                    print(
                        f"Epoch: {epoch} batch: {batch_idx + 1} train loss: {loss.item():.5f}"
                    )
                    self.save_model(save_dir=save_dir, epoch=epoch)

                # loss.backward()
                # Scale gradients and perform backprop
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.llm.parameters(), max_norm=1.0)

                # self.optimizer.step()
                # Update optimizer and scale the loss scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

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
                print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")
            self.llm.train()  # Restore to training mode

            self.scheduler.step()

        # Training Finished, cleaning up
        torch.utils.data.distributed.destroy_process_group()

    def model_summary(self):
        print(summary(self.llm, input_size=(1, self.embed_dim), dtypes=[torch.int32]))

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
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.llm.module.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            return start_epoch
        else:
            return 1

    def save_model(self, save_dir, epoch):
        if self.rank == 0:
            # Save model after each epoch
            save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.llm.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                },
                save_path,
            )
            print(f"Saved model {save_path}")

    def generate_text(self, prompt, max_len=50):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        ).to(self.device)
        input_tokens = inputs["input_ids"]

        generated_ids = []
        for _ in range(max_len):
            with torch.no_grad():
                outputs = self.llm(input_tokens)
                next_token_logits = outputs[:, -1, :]  # Last token's logits
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated_ids.append(next_token_id)
                input_tokens = torch.cat([input_tokens, next_token_id], dim=1)

        generated_text = self.tokenizer.decode(
            torch.cat(generated_ids, dim=1)[0], skip_special_tokens=True
        )
        return generated_text
