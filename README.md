# Heart Disease Predictions

This repository contains the code for a personal machine learning project aiming to **predict the risk of heart disease** in American patients using health survey data.  

The project is based on the [*Indicators of Heart Disease (2022 update)*](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) dataset, which compiles responses from more than **400,000 adults** to a CDC health survey conducted in the USA.

---

## Project Description

This project explores the end-to-end process of building a **predictive model** from raw tabular data.  
It includes the following stages:

1. **Data cleaning and preprocessing**  
   - Handling missing values and outliers  
   - Encoding categorical features and scaling numerical data  

2. **Feature engineering and selection**  
   - Polynomial feature generation and correlation analysis  
   - Selection of the most informative features for prediction  

3. **Modeling and evaluation**  
   - Comparison of multiple algorithms (e.g., LightGBM, RandomForest, Logistic Regression, etc.)  
   - Performance tracking and experiment management using **MLflow**  

4. **Hyperparameter optimization**  
   - Fine-tuning of model parameters using **Optuna** for efficient search  

5. **Explainability**  
   - Global and local interpretation of the best model using **SHAP (Shapley values)**  

The final goal is to combine **robust predictive performance** with **interpretability**, to better understand which health factors most influence the risk of heart disease.

---

## Repository Structure

```
├── main.ipynb # Jupyter notebook with the complete pipeline
├── src/
│ └── utils.py # Utility functions for data processing, modeling, and evaluation
├── data/
│ └── (not included) # Placeholder for dataset (see below)
├── mlruns/ # MLflow experiment tracking directory
├── heart_disease.db # Local SQLite MLflow tracking database
└── README.md
```

---

## Data Source

The dataset used in this project is **publicly available on Kaggle**:

> [Indicators of Heart Disease (2022 update)](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

To reproduce the analysis:
1. Download the dataset from Kaggle.
2. Place the CSV file(s) into the `data/` folder.
3. Update the data path in `main.ipynb` if necessary.

---

## Dependencies

The project relies on the following Python packages (Python ≥ 3.10 recommended):

- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `lightgbm`, `optuna`, `mlflow`, `shap`

You can install them with:

```bash
pip install -r requirements.txt
```

---

## How to run
1. Clone the repository:
```bash
git clone https://github.com/tazziotissot/Heart-disease-prediction.git
cd Heart-disease-prediction
```
2. Launch Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```
3. Optionally, start MLflow to visualize experiment tracking:
```bash
mlflow ui --backend-store-uri sqlite:///heart_disease.db
```

---
## Results and Insights
### Model Comparison
Several models were trained and evaluated using k-fold stratified cross-validation, including logistic regression, random forest, and gradient boosting. The comparison showed that gradient boosting models provided the best balance between recall and precision, making it the final model retained for analysis.

### Hyperparameter Optimization
Hyperparameter tuning was conducted using Optuna, allowing the model to reach a more optimal balance between bias and variance. A tree parzen estimator sampling strategy explored parameters such as the number of leaves, maximum depth, learning rate, and regularization strength. The tuned LightGBM model demonstrated improved generalization performance compared to its default configuration.

### Explainability (SHAP Analysis)
Using SHAP (Shapley Additive Explanations), the project identified the most influential features contributing to heart disease risk prediction. The most impactful variables included age, physical health, gender, smoking habits, and the presence of previous medical conditions. Individual-level SHAP analyses provided insights into how each factor influenced predictions for specific patients.

---

## Key Takeaways
- Complete end-to-end pipeline: from data cleaning to model explainability
- Experiment tracking and reproducibility ensured via MLflow
- Automated hyperparameter search using Optuna
- Emphasis on model transparency and interpretability with SHAP
- Gradient boosting selected as the final modelling approach for its strong performance and interpretability

----

## Future Improvements
- Extend the analysis to include calibration and threshold optimization for clinical applicability
- Explore ensemble approaches combining tree-based and linear models
- Integrate a web dashboard to visualize predictions and SHAP explanations interactively
- Evaluate model fairness across demographic subgroups