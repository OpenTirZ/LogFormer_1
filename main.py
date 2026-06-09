import torch
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset , DataLoader
import math


from Data.data import GettingData,makingEventToeventId,makingVocab
from Dataloader.dataloader import create_dataloader
from Attention.MultiHeadAttention import MultiHeadAttention
from GELU.GELU import GELU
from LayerNorm.LayerNorm import LayerNorm
from Transformer.Transformer import TransformerBlock
from Testing.test import calculate_anomaly_scores , plot

TirZ_LogFormer = {
    "vocab_size": 14,
    "context_length": 10,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "model_type": "gpt"
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # The model outputs logits for each token in the sequence (batch_size, seq_length, vocab_size)
    # The dataset currently provides a single target token per sequence (batch_size,)
    # We will compute the loss based on the prediction for the *last* token in the input sequence.
    logits = model(input_batch) # Shape: (batch_size, seq_length, vocab_size)

    # Select logits corresponding to the last token in the sequence
    # Shape: (batch_size, vocab_size)
    logits_for_next_token = logits[:, -1, :]

    # Compute cross-entropy loss with the single target token
    loss = torch.nn.functional.cross_entropy(logits_for_next_token, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Open a file to log training status
    with open('training_log.txt', 'w') as f:
        f.write('Epoch,Step,Train Loss,Val Loss\n') # Write header

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    log_message = (f"Ep {epoch+1} (Step {global_step:06d}): "
                                   f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                    print(log_message)
                    # Write training status to file
                    f.write(f"{epoch+1},{global_step},{train_loss:.3f},{val_loss:.3f}\n")

    return train_losses, val_losses, track_tokens_seen

import torch
import math
import pandas as pd

def calculate_anomaly_scores(
    model,
    events,
    event_to_id,
    context_length=10
):
    model.eval()

    results = []

    with torch.no_grad():

        for idx in range(1, len(events)):

            # Build context
            start = max(0, idx - context_length)

            context_events = events[start:idx]

            actual_event = events[idx]

            context_ids = [
                event_to_id[e]
                for e in context_events
            ]

            actual_id = event_to_id[actual_event]

            x = torch.tensor(
                context_ids,
                dtype=torch.long
            ).unsqueeze(0)

            probs = torch.softmax(
                model(x),
                dim=-1
            )

            last_probs = probs[0, -1]

            actual_prob = (
                last_probs[actual_id]
                .item()
            )

            anomaly_score = -math.log(
                actual_prob + 1e-10
            )

            results.append({
                "position": idx,
                "context": " ".join(context_events),
                "actual_event": actual_event,
                "probability": actual_prob,
                "anomaly_score": anomaly_score
            })

    return pd.DataFrame(results)

all_event_ids_list = GettingData()
event_template_to_id = makingEventToeventId()

train_ratio = 0.9
split = int(len(all_event_ids_list) * train_ratio)

train_data = all_event_ids_list[:split]
val_data = all_event_ids_list[split:]

train_dataloader = create_dataloader(train_data , vocab , batch_size=16)
val_dataloader = create_dataloader(val_data , vocab , batch_size=16)

print(train_dataloader.dataset.input_ids)
print(train_dataloader.dataset.target_ids)


class TirZ_LogFormer_L1(nn.Module):
    def __init__(self , cfg) :
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"] , cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"] , cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.norm_leayer = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"] , cfg["vocab_size"] , bias=False
        )

    def forward(self, x):
        batch_size , seq_length = x.shape

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_length , device=x.device))

        x = tok_emb + pos_emb

        x = self.drop_emb(x)
        x = self.trf_blocks(x)

        x = self.norm_leayer(x)

        x = self.out_head(x)
        return x

torch.manual_seed(123)
model = TirZ_LogFormer_L1(TirZ_LogFormer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure model is on the correct device
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000004, weight_decay=0.1)

num_epochs = 5 # You can adjust this as needed
eval_freq = 5   # Evaluate every 5 steps
eval_iter = 5   # Use 5 batches for evaluation

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_dataloader, val_dataloader, optimizer, device,
    num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter
)

# Testing 
events = ['E1', 'E3', 'E7', 'E2', 'E5', 'E11']
df_scores = calculate_anomaly_scores(
    model,
    events,
    vocab, # Corrected: use 'vocab' which maps EventId to numerical indices
    context_length=10
)

plot(df_scores)