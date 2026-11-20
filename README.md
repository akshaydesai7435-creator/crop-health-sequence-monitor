# Crop Health Sequence Monitor

Crop Health Sequence Monitor for tracking agricultural observations (crop diseases, pests, soil moisture, and interventions), mining frequent co-occurring patterns, and predicting the next agricultural item using a fine-tuned language model.

## Installation

```powershell
pip install -r requirements.txt
```

## Usage

### 1. Mine Frequent Patterns
Display frequent itemsets from agricultural sequences:
```powershell
python -m app.cli patterns --min-support 0.1 --max-size 3
```

### 2. Train Language Model
Continue pretraining the small language model on agricultural sequence baskets:
```powershell
python -m app.cli train --epochs 5 --out-dir models/crop_health_lm
```

### 3. Evaluate Predictions
Evaluate next-item predictions on unseen test sequences:
```powershell
python -m app.cli evaluate --model-dir models/crop_health_lm
```

### 4. Predict Next Item via CLI
Query the next likely agricultural observation or action given current observations:
```powershell
python -m app.cli predict "Aphid Infestation" "Leaf Curling" --model-dir models/crop_health_lm
```

## Running Tests

Run the test suite with pytest:
```powershell
python -m pytest -q
```
