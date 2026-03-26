# Gold Price Dynamics Analysis

A data science project analyzing gold price trends and building predictive models.

## Project Structure

```
gold-price-analysis/
├── data/               # Dataset folder
├── notebooks/          # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data_processing.py
│   ├── visualization.py
│   └── model.py
├── models/            # Saved models
├── results/           # Analysis outputs
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Getting Started

### 1. Download the Dataset

1. Go to https://www.kaggle.com/datasets/krupalpatel07/gold-price-dynamics
2. Download the dataset
3. Place the CSV file in the `data/` folder

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Analysis

Open `notebooks/01_exploratory_analysis.ipynb` in Jupyter and follow along!

## Key Analyses

- Price trend visualization
- Seasonal patterns
- Correlation analysis
- Price prediction using ML

## Results

Findings and visualizations are saved in the `results/` folder.
