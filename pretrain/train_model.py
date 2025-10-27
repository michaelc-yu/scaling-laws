import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
import os

from model.factory import build_model
from pretrain.dataset import build_causal_lm_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_sizes.yaml")
    parser.add_argument("--size", type=str, default="300M")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default="data/processed/wikipedia_deduped.jsonl")
    parser.add_argument("--manifest", type=str, default="data/splits/pretrain_manifest.jsonl")
    parser.add_argument("--num_proc", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Building {args.size} model...")
    model = build_model(args.config, args.size)
    model.to(device)
    print(f"model: {model}")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    print("Loading dataset...")
    train_ds, tokenizer = build_causal_lm_dataset(
        full_data_path=args.data_dir,
        manifest_path=args.manifest,
        tokenizer_name=args.tokenizer,
        block_size=args.block_size,
        num_proc=args.num_proc,
    )
    print(f"type of dataset: {type(train_ds)}")
    print(train_ds)
    print(train_ds.column_names)
    print(train_ds[0])
    print(train_ds[:3])

    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    os.makedirs(args.save_dir, exist_ok=True)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done")
        print(f"Avg loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(args.save_dir, f"{args.size}_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
