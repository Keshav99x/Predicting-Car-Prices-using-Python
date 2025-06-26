
# A Basic Car Price Prediction using Random Forest Regressor

This project builds a machine learning model to predict car prices based on various features such as brand, fuel type, mileage, and other technical attributes. The model uses the `RandomForestRegressor` algorithm and is implemented in a Jupyter Notebook using Python.

## 📊 Dataset

The dataset used is `Car_prices_known.csv`, which was extracted from kaggle:

The following features were used for prediction:
- `Price` – Target variable (car price in currency units)
-  `Engine volume`, `Mileage`, `Cylinders`, `Airbags` – Numerical features
- `Prod. year`, `Fuel type`, `Leather interior`,  `Gear box type`, `Drive wheels` – Categorical features ( encoded)
- `Car_age` – Derived feature (2025 - `Prod. year`)

## 🧰 Tools & Libraries Used

- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib** – Visualization
- **Scikit-learn** – Machine learning (`train_test_split`, `RandomForestRegressor`, evaluation metrics)

## 🧪 Model Training and Evaluation

- The dataset was split into training and test sets (80/20 split).
- A **RandomForestRegressor** with 100 estimators was trained on the dataset.
- The model achieved an **R-squared value of 0.93** on training data and **0.80** on test data.
- A scatter plot shows a strong correlation between actual and predicted car prices.

## 📈 Results

- The R² score on training set: `0.933`
- The R² score on test set: `0.805`
- The actual vs predicted price plot shows the model generalizes well but may slightly underfit at higher price ranges.
