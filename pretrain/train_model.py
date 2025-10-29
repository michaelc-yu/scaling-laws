import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
import os
import yaml

from model.factory import build_model
from pretrain.dataset import build_causal_lm_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model_sizes.yaml")
    parser.add_argument("--training_config", type=str, default="configs/training.yaml")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml")
    parser.add_argument("--size", type=str, default="tiny")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)["models"][args.size]
    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)["training"]
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)["data"]

    model_sz = args.size
    lr = float(train_cfg["lr"])
    block_sz = train_cfg["block_size"]
    batch_sz = train_cfg["batch_size"]
    save_dir = train_cfg["save_dir"]
    num_epochs = train_cfg["epochs"]

    full_data_path = data_cfg["full_data_path"]
    manifest_path = data_cfg["manifest_path"]
    tokenizer_name = data_cfg["tokenizer_name"]

    num_proc = data_cfg.get("num_proc", None)


    print(f"Building {model_sz} model...")
    model = build_model(model_cfg, model_sz)
    model.to(device)
    print(f"model: {model}")

    optimizer = AdamW(model.parameters(), lr=lr)

    print("Loading dataset...")
    train_ds, tokenizer = build_causal_lm_dataset(
        full_data_path=full_data_path,
        manifest_path=manifest_path,
        tokenizer_name=tokenizer_name,
        block_size=block_sz,
        num_proc=num_proc,
    )
    print(f"Dataset type: {type(train_ds)}")
    print(train_ds)
    print(train_ds.column_names)
    print(train_ds[0])
    print(train_ds[:3])

    dataloader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

    os.makedirs(save_dir, exist_ok=True)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

        ckpt_path = os.path.join(save_dir, f"{model_sz}_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
