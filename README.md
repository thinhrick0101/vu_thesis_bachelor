# Transformer Attention Analysis

This repository contains code and models for analyzing and comparing dense and sparse transformer architectures at the byte/character level, with a focus on attention head interpretability and efficiency. The project includes training scripts, model definitions, and analysis pipelines for extracting and visualizing attention head metrics.

## Repository Structure

```
vu_thesis/
│
├── dense_transformer/
│   ├── stable_char_transformer.py
│   └── train_dense_model.py
│
├── sparse_transformer/
│   ├── sparse_attention.py
│   ├── sparse_byte_transformer.py
│   ├── stable_char_transformer.py
│   └── train_sparse_transformer.py
│
├── pipeline/
│   ├── extract_head_metrics.py
│   ├── head_metrics.csv
│   └── make_silhouette_plot_thesis.py

### Folder Descriptions

- **dense_transformer/**  
  Contains code for the dense (standard) transformer model and its training script.
  - `stable_char_transformer.py`: Implementation of the dense character-level transformer model.
  - `train_dense_model.py`: Script to train the dense transformer on the Enwik8 dataset.

- **sparse_transformer/**  
  Contains code for the sparse transformer model, which uses custom sparse attention patterns, and its training script.
  - `sparse_attention.py`: Implementation of custom sparse multi-head attention mechanisms.
  - `sparse_byte_transformer.py`: Byte-level transformer model using sparse attention.
  - `stable_char_transformer.py`: (Likely) an alternative or compatible implementation for sparse attention.
  - `train_sparse_transformer.py`: Script to train the sparse transformer on the Enwik8 dataset.

- **pipeline/**  
  Contains scripts for analyzing trained models, extracting attention head metrics, and visualizing results.
  - `extract_head_metrics.py`: Extracts per-head attention metrics (entropy, sparsity, distance) from a trained model and saves them to `head_metrics.csv`.
  - `head_metrics.csv`: Output file containing extracted attention head metrics.
  - `make_silhouette_plot_thesis.py`: Performs clustering analysis on attention head metrics and generates silhouette plots for thesis figures.



---

## Getting Started

### 1. Environment Setup

Install the required Python packages (PyTorch, numpy, pandas, scikit-learn, matplotlib, wandb, etc.).  
You may want to use a virtual environment.

```bash
pip install torch numpy pandas scikit-learn matplotlib wandb
```

### 2. Training a Model

#### Dense Transformer

```bash
cd dense_transformer
python train_dense_model.py --num_epochs 20 --batch_size 32 --seq_length 1024
```

#### Sparse Transformer

```bash
cd sparse_transformer
python train_sparse_transformer.py --num_epochs 20 --batch_size 32 --seq_length 1024
```

Both scripts will download the Enwik8 dataset if not present, train the model, and save the checkpoint in the `models/` directory.

### 3. Extracting Attention Head Metrics

After training, extract attention head metrics for analysis:

```bash
cd pipeline
python extract_head_metrics.py --model_path ../models/dense_char_transformer.pt --output head_metrics.csv
```

- You can also analyze the sparse model by changing the `--model_path`.

### 4. Visualizing Attention Head Clusters

Generate silhouette plots to analyze the clustering of attention heads:

```bash
python make_silhouette_plot_thesis.py
```

This will read `head_metrics.csv` and produce a silhouette analysis plot for your thesis.

---

## Notes

- The code is designed for research and analysis of transformer attention patterns, especially for interpretability studies.
- The `wandb` integration is used for experiment tracking and visualization.
- The models are trained on the Enwik8 dataset (byte-level language modeling).

---

## Citation

If you use this codebase in your research, please cite appropriately. 