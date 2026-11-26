# Lightweight ML-based Intrusion Detection for IoT Networks

Compares traditional ML classifiers with a feature-selected, lightweight Random Forest
on IoT network-flow data, measuring accuracy, F1, inference time and model size.

## Setup
```bash
pip install -r requirements.txt
```

## Data
Download a public IoT flow dataset (e.g. CIC-IoT2023 or ToN-IoT) and place the CSV file(s) in `data/`.
The dataset is not included in this repo.

## Run
```bash
# Benign vs Attack
python train.py --data data/ --label-col label --binary --sample 200000
# Multi-class
python train.py --data data/ --label-col label --sample 200000
```

## Outputs (`results/`)
`results.csv`, `summary.json`, `confusion_matrix.png`, `feature_importance.png`
