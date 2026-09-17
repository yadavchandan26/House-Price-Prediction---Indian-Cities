# House Price Prediction (India)

A small end-to-end **training pipeline** that predicts house prices in a handful of Indian cities using a simple Linear Regression model. Built as a machine learning practice project to work through the standard workflow: load → preprocess → split → train → evaluate → predict.

## Project Structure

```
house_price_india/
├── main.py                          # Entry point — runs the full pipeline
├── data/
│   └── dataset.csv                  # Raw housing data
├── prediction/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Loading, cleaning, feature engineering, train/test split
│   ├── model_training.py            # Model definition and fitting
│   └── model_evaluation.py          # Scoring and submission file generation
└── output.csv                       # Generated predictions (created after running)
```

## Dataset

Each row represents a property with the following columns:

| Column          | Description                                  |
|-----------------|-----------------------------------------------|
| `Area_SqFt`     | Property size in square feet                  |
| `Rooms`         | Number of rooms                               |
| `Build_Year`    | Year the property was built                   |
| `Location`      | City (Delhi, Gurugram, Noida, Lucknow, Jaipur, Indore, Kanpur, Prayagraj) |
| `Street_Type`   | Street/road category                          |
| `Furnishing`    | Furnishing status                             |
| `Property_Type` | Apartment / Duplex / Villa etc.                |
| `Has_Pool`      | Yes/No                                        |
| `Price`         | Target variable                               |

## Pipeline Overview

1. **Load & clean** — reads the CSV and drops rows with missing values.
2. **Feature engineering**
   - `Age` is derived from `Build_Year` (reference year: 2026).
   - `Location` is manually mapped to a numeric score (`location_hack`) based on rough city tier.
   - `Has_Pool` is one-hot encoded.
3. **Feature selection** — final feature set: `Area_SqFt`, `Rooms`, `Age`, `Has_Pool`, `location_hack`.
4. **Train/test split** — 70/30 split, `random_state=42`.
5. **Training** — `sklearn.linear_model.LinearRegression`.
6. **Evaluation** — R² score.
7. **Output** — predictions on the test set written to `output.csv`.

## Usage

```bash
python main.py
```

This trains the model on `data/dataset.csv` and writes predictions to `output.csv` in the project root.

## Requirements

```
pandas
scikit-learn
```

## Known Limitations

This is a practice/learning project, not a production pipeline. Some intentional simplifications worth knowing about:

- **No feature scaling** — `Area_SqFt` and `Rooms` are on very different scales, which affects linear regression's coefficients.
- **`location_hack` is an ordinal hack** — `Location` is a categorical variable with no natural order; a proper approach would one-hot encode it like `Has_Pool`.
- **Evaluation runs on the training set**, not the held-out test set, so the reported score is not a true measure of generalization.
- **No model persistence** — the trained model isn't saved (e.g. via `joblib`), so it can't be reused without retraining.
- **No input validation** — an unexpected `Location` value not covered by the mapping will silently produce `NaN` and break training.
- **Hardcoded paths/params** — data path, split ratio, and random seed are fixed in code rather than configurable.
