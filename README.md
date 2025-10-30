
# Scaling-Laws


Scaling-laws pretraining vs. RAG


### 1: Setup a virtual environment (ideally with python 3.11) and activate it
```bash
# (preferred)
python3.11 -m venv venv

# (can also do this but must use a supported python version)
python -m venv venv

source venv/bin/activate
```
### 2: Install requirements.txt
```bash
pip install -r requirements.txt
```
### 3: Prepare the wikipedia data
Modify configs/data_prep.yaml to desired configs.
Currently only wikipedia dataset is supported.
```bash
chmod +x scripts/run_data_prep.sh
bash scripts/run_data_prep.sh
```
This will download num_tokens tokens from wikipedia and save the raw jsonl to output_path. It will then run dedupe on the dataset and save to deduped_data path. Then it will split the data into the pretrain portion and retrieval portion based on pretrain_frac and save to pretrain_manifest and retrieval_manifest. Pretraining will read from pretrain_manifest and RAG will read from retrieval_manifest.

### 4: Build the RAG index
This will build a retrieval index using FAISS + BM25 hybrid from only the retrieval tokens partition of the split (reads only from retrieval_manifest).
```bash
chmod +x scripts/run_build_index.sh
bash scripts/run_build_index.sh
```
### 5: Login to wandb
```bash
python -m wandb login
```

### 6: Run experiments by setting desired parameters in pretrain/train_experiment_driver.py and executing it

Edit EXPERIMENTS list in pretrain/train_experiment_driver.py then run:
```bash
python -m pretrain.train_experiment_driver
```
This will run all parameter combinations and execute train_model on each of them. Hopefully easier to track training runs this way.


#### 6.5 (Optional): Can also train by directly reading from yaml file but this shouldn't be needed since above step can do all this easier

Modify configs/data.yaml, configs/model_sizes.yaml, and configs/training.yaml. Then run training with
```bash
python -m pretrain.train_model
```

### 7: Download and prepare the eval datasets
This will download and prepare 3 datasets (GSM8K, StrategyQA, and SVAMP)
```bash
chmod +x scripts/prepare_all_datasets.sh
bash scripts/prepare_all_datasets.sh
```

### 8: Evaluation and everything else (WIP)


### Note on Running Scripts

This repo uses Python's **module-based execution** to ensure proper relative imports.

Always run scripts using:
python -m model.train

Avoid running with:
python model/train.py


### Libraries
Using the einops library for tensor operations (https://einops.rocks/)


