# Personalized ECG Signal Classification using Transfer Learning

A deep learning project that builds a general CNN based ECG classifier and personalizes it per patient using transfer learning (fine tuning)

## Project Structure

```
ECG_Project/
├── data/                    # Downloaded ECG data stored here
├── preprocessing.py         # Data loading, segmentation, normalization
├── model.py                 # CNN architecture
├── train.py                 # Train general model on multiple patient data
├── personalize.py           # Fine tuned the model per patient
├── evaluate.py              # Metrics, confusion matrix, comparison plots
├── results/
│   └── figures/             # Results 
├── requirements.txt
└── README.md
```

## Objectives

 General Model – Train a CNN on multi patient MIT BIH ECG data
 Personalization – Fine tune on individual patient data 5%, 10%, 20%
 Variability Analysis – Study inter patient ECG differences
 Data Efficiency – Find minimum data needed for fine tuning
 Comparison – General vs personalized model metrics

## Setup & Installation
### Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Steps

### 1. Download and preprocess data
```bash
python preprocessing.py
```
Downloads MIT BIH Arrhythmia Database via wfdb, segments beats, normalizes signals, and saves processed data

### 2. Train general model
```bash
python train.py
```
Trains CNN on combined multi patient data and saves model to results/general_model.keras

### 3. Personalize per patient
```bash
python personalize.py
```
Fine tunes general model for each patient at 5%, 10%, 20% data fractions

### 4. Evaluate and compare
```bash
python evaluate.py
```
Generates comparison charts and saves all figures to results/figures/

## Expected results

| Model Type          | Accuracy | F1 Score |
|---------------------|----------|----------|
| General Model       | ~85-88%  | ~0.83    |
| Personalized (5%)   | ~89-91%  | ~0.88    |
| Personalized (20%)  | ~93-96%  | ~0.93    |

## Dataset

MIT BIH Arrhythmia Database (automatically downloaded via wfdb)
- 48 half hour ECG recordings, 360 Hz sampling rate
- Labels- Normal (N) and Abnormal (V, A, L, R, etc.)

## Key Design Decisions

 Patients are strictly separated in train/test splits
 Class imbalance handled with class weights
 Fine tuning uses only a fraction of patient data

## Technologies Used

Python 3.9+ | TensorFlow/Keras | WFDB | NumPy | Scikit-learn | Matplotlib | Seaborn
