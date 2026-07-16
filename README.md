# Alzheimer's Disease Classification from FDG-PET using Attention-Guided Deep Learning & XAI

## Project Structure
```
alzheimers_pet_xai/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters & paths
├── data/
│   ├── dataset.py             # ADNI + OASIS-3 dataset loader
│   └── preprocessing.py       # Preprocessing pipeline
├── models/
│   ├── cbam.py                # CBAM attention module
│   └── resnet_cbam.py         # ResNet50 + CBAM model
├── train.py                   # Training loop
├── evaluate.py                # Evaluation & metrics
├── xai/
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── shap_explainer.py      # SHAP DeepExplainer
│   └── faithfulness.py        # Insertion / Deletion AUC
├── utils/
│   ├── metrics.py             # F1, AUC-ROC, Kappa
│   └── visualize.py           # Heatmap overlays & plots
└── main.py                    # Entry point
```

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
# Train
python main.py --mode train

# Evaluate
python main.py --mode evaluate --checkpoint checkpoints/best_model.pth

# Generate XAI
python main.py --mode xai --checkpoint checkpoints/best_model.pth
```
