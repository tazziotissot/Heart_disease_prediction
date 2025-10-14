import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Activation nécessaire
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
import lightgbm as lgb
import optuna
import shap
import mlflow


def tukey(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Applies Tukey's method to detect and replace outliers in a specified column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name on which to apply Tukey's outlier detection method.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with outliers replaced by NaN values.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iq = q3 - q1
    mask = (df[col] > q3 + 1.5 * iq) | (df[col] < q1 - 1.5 * iq)
    df.loc[mask, col] = np.nan
    return df.copy()


def convert_binary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Converts a binary categorical column into numerical format (0/1).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Name of the column to convert.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the binary column converted to numeric.
    """
    if "nan" in df[col].unique():
        df.loc[df[col] == "nan", col] = np.nan
    if df[col].nunique() == 2 and "Yes" in df[col].unique():
        df[col] = np.where(df[col] == "Yes", 1, 0)
    return df.copy()


def create_polynomial(df: pd.DataFrame, col: str, power_max: int = 3) -> pd.DataFrame:
    """
    Creates polynomial feature expansions for a numerical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to transform.
    power_max : int, optional (default=3)
        Maximum polynomial power to generate.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame including new polynomial columns.
    """
    if df[col].nunique() > 2:
        for i in range(2, power_max + 1):
            df[f"{col}_exp{i}"] = df[col] ** i
    return df.copy()


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment: str,
    k: int = 5,
    early_stopping_rounds: int = 10,
    name: str | None = None,
    tracking_uri: str = "sqlite:///heart_disease.db",
) -> dict:
    """
    Evaluates a machine learning model using cross-validation, logs results to MLflow, and returns metrics.

    Parameters
    ----------
    model : estimator
        Model to evaluate (supports LightGBM and scikit-learn compatible models).
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    experiment : str
        MLflow experiment name.
    k : int, optional (default=5)
        Number of cross-validation folds.
    early_stopping_rounds : int, optional (default=10)
        Number of early stopping rounds for LightGBM.
    name : str, optional
        Model name for MLflow logging.
    tracking_uri : str, optional
        MLflow tracking URI.

    Returns
    -------
    dict
        Dictionary containing cross-validation metrics, test metrics, and model name.
    """
    # Definition of evaluation metrics
    scoring_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    mlflow.set_tracking_uri(tracking_uri)
    current_experiment = mlflow.set_experiment(experiment)
    with mlflow.start_run():

        # If the model is LightGBM
        if isinstance(model, lgb.LGBMClassifier):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            val_scores = {metric: [] for metric in scoring_metrics}
            for train_idx, val_idx in skf.split(X_train, y_train):
                # Splitting the training/validation folds
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # Training with early stopping
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(early_stopping_rounds)],
                )

                # Prediction on the validation fold
                y_pred_val = model.predict(X_val)
                y_proba_val = model.predict_proba(X_val)[:, 1]

                # Storing the scores
                val_scores["accuracy"].append(accuracy_score(y_val, y_pred_val))
                val_scores["precision"].append(precision_score(y_val, y_pred_val))
                val_scores["recall"].append(recall_score(y_val, y_pred_val))
                val_scores["f1"].append(f1_score(y_val, y_pred_val))
                val_scores["roc_auc"].append(roc_auc_score(y_val, y_proba_val))
                val_scores["average_precision"].append(
                    average_precision_score(y_val, y_proba_val)
                )

            # Mean and standard-deviation of validation scores for LightGBM
            cv_summary = {
                f"cv_{metric}_mean": np.mean(val_scores[metric])
                for metric in scoring_metrics
            }
            cv_summary.update(
                {
                    f"cv_{metric}_std": np.std(val_scores[metric])
                    for metric in scoring_metrics
                }
            )
            # Final training on the whole training set
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(early_stopping_rounds)],
            )

        else:
            # Regular cross-validation process for other scikit-learn models
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=skf,
                scoring=scoring_metrics,
                return_train_score=False,
            )

            # Mean and standard-deviation of validation scores
            cv_summary = {
                f"cv_{metric}_mean": np.mean(cv_results[f"test_{metric}"])
                for metric in scoring_metrics
            }
            cv_summary.update(
                {
                    f"cv_{metric}_std": np.std(cv_results[f"test_{metric}"])
                    for metric in scoring_metrics
                }
            )
            # Final training on the whole training set
            model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = 1 / (1 + np.exp(-model.decision_function(X_test)))
        else:
            y_proba = None

        # Computing metrics on the test set
        performance_test = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }

        # Computing the ROC AUC if and only if the model class has support for the predict_proba function
        if y_proba is not None:
            performance_test["test_roc_auc"] = roc_auc_score(y_test, y_proba)
            performance_test["test_average_precision"] = average_precision_score(
                y_test, y_proba
            )
        # Logging the run parameters and results
        mlflow.log_params({"model": name})
        mlflow.log_metrics({**cv_summary, **performance_test})
        if isinstance(model, lgb.LGBMClassifier):
            mlflow.lightgbm.log_model(
                lgb_model=model,
                input_example=X_test.astype("float"),
                artifact_path=f"{name}_trained_model",
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                input_example=X_test.astype("float"),
                artifact_path=f"{name}_trained_model",
            )

    # Return scores as a dictionary
    return {**cv_summary, **performance_test, "model": name}


def impute(
    df: pd.DataFrame, df_test: pd.DataFrame, strategy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputes missing values in training and test DataFrames using different imputation strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset.
    df_test : pd.DataFrame
        Test dataset.
    strategy : str
        Imputation strategy ('simple' or 'iterative').

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple containing the imputed training and test DataFrames.
    """
    df_copy = df.copy()
    df_test_copy = df_test.copy()
    cat_columns = df_copy.select_dtypes("O").columns.tolist()
    num_columns = [
        col
        for col in df_copy.select_dtypes(["int", "float"]).columns
        if df_copy[col].nunique() > 2
    ]
    remaining_columns = [
        col
        for col in df_copy.columns
        if col not in cat_columns and col not in num_columns
    ]
    former_dtypes = {c: str(df_copy[c].dtype) for c in df_copy.columns}

    # Imputation
    if strategy == "simple":
        column_transformer = ColumnTransformer(
            transformers=[
                (
                    "median",
                    make_pipeline(
                        StandardScaler(), SimpleImputer(strategy="median"), num_columns
                    ),
                ),
                ("mode", SimpleImputer(strategy="most_frequent"), cat_columns),
            ],
            remainder="passthrough",
        )
    elif strategy == "iterative":
        column_transformer = ColumnTransformer(
            transformers=[
                (
                    "iterative",
                    make_pipeline(
                        StandardScaler(),
                        IterativeImputer(max_iter=30, random_state=42, verbose=2),
                    ),
                    num_columns,
                ),
                ("mode", SimpleImputer(strategy="most_frequent"), cat_columns),
            ],
            remainder="passthrough",
        )
    else:
        raise ValueError(f"Strategy {strategy} unknown")
    df_copy = pd.DataFrame(
        column_transformer.fit_transform(df_copy),
        columns=num_columns + cat_columns + remaining_columns,
    )
    df_test_copy = pd.DataFrame(
        column_transformer.transform(df_test_copy),
        columns=num_columns + cat_columns + remaining_columns,
    )
    df_copy = df_copy.astype(former_dtypes)
    df_test_copy = df_test_copy.astype(former_dtypes)

    return df_copy, df_test_copy


def plot_model_performance(results_list: list[dict]) -> None:
    """
    Plots a comparative bar chart of model performance metrics with error bars.

    Parameters
    ----------
    results_list : list of dict
        List of dictionaries containing performance metrics for each model.

    Returns
    -------
    None
        Displays a matplotlib bar plot comparing model metrics.
    """
    metrics = {
        "cv_precision_mean": "Precision",
        "cv_recall_mean": "Recall",
        "cv_f1_mean": "F1 Score",
        "cv_average_precision_mean": "PR AUC",
    }
    errors = {
        "cv_precision_std": "Precision",
        "cv_recall_std": "Recall",
        "cv_f1_std": "F1 Score",
        "cv_average_precision_std": "PR AUC",
    }

    df = pd.DataFrame(results_list)

    df_metrics = df[["model"] + list(metrics.keys())]
    df_metrics = df_metrics.rename(columns=metrics)
    df_metrics = df_metrics.drop_duplicates()

    df_errors = df[["model"] + list(errors.keys())]
    df_errors = df_errors.rename(columns=errors)
    df_errors = df_errors.drop_duplicates()

    df_melted = df_metrics.melt(
        id_vars=["model"], var_name="Metric", value_name="Score"
    )
    df_errors_melted = df_errors.melt(
        id_vars=["model"], var_name="Metric", value_name="Std"
    )
    df_final = df_melted.merge(df_errors_melted, on=["model", "Metric"], how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = df_final["model"].unique()
    x = np.arange(len(models))
    width = 0.2
    for i, metric in enumerate(df_final["Metric"].unique()):
        subset = df_final[df_final["Metric"] == metric]
        ax.bar(
            x + i * width,
            subset["Score"],
            width=width,
            yerr=subset["Std"],
            capsize=5,
            label=metric,
        )
    ax.set_xticks(x + width, models, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of Model Performances")
    ax.legend()


def gradient_boosting_focal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    experiment: str,
    k: int = 5,
    early_stopping_rounds: int = 10,
    tracking_uri: str = "sqlite:///heart_disease.db",
) -> tuple[float, float]:
    """
    Trains LightGBM models using a custom focal loss function and logs the results to MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    experiment : str
        MLflow experiment name.
    k : int, optional (default=5)
        Number of cross-validation folds.
    early_stopping_rounds : int, optional (default=10)
        Early stopping parameter for LightGBM.
    tracking_uri : str, optional
        MLflow tracking URI.

    Returns
    -------
    (float, float)
        Mean and standard deviation of the average precision score across folds.
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def focal_loss(y_pred, df, gamma=2.0, alpha=0.25):
        y_true = df.get_label()
        p = sigmoid(y_pred)
        grad = alpha * (1 - p) ** gamma * (y_true - p)  # Gradient
        hess = (
            alpha * (1 - p) ** gamma * p * (1 - p) * (gamma * (y_true - p) + 1)
        )  # Hessian
        return grad, hess

    mlflow.set_tracking_uri(tracking_uri)
    current_experiment = mlflow.set_experiment(experiment)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    val_scores = []
    for idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        with mlflow.start_run() as run:
            # Splitting of training/validation folds
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            # Training with early stopping
            model = lgb.train(
                params={
                    "objective": lambda y_pred, df: focal_loss(
                        y_pred, df, 0.0, 1 / y_tr.mean()
                    ),
                    "metric": "auc",
                    "n_jobs": -2,
                },  # We define a binary base
                train_set=lgb.Dataset(X_tr, label=y_tr),
                valid_sets=[lgb.Dataset(X_val, label=y_val)],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(10)],
            )
            y_pred = sigmoid(model.predict(X_val))
            val_scores.append(average_precision_score(y_val, y_pred))
            mlflow.log_params({"model": "Boost", "loss": "focal", "fold": idx})
            mlflow.log_metric("average_precision", val_scores[-1])
    return np.mean(val_scores), np.std(val_scores)


def hyperparameterize(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    experiment: str,
    k: int,
    early_stopping_rounds: int = 10,
    default_params: dict = {},
    params_to_optimize: dict = {
        "num_leaves": [10, 100],
        "max_depth": [3, 12],
        "learning_rate": [1e-3, 0.3],
        "min_child_samples": [5, 100],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "reg_alpha": [1e-8, 10.0],
        "reg_lambda": [1e-8, 10.0],
    },
    tracking_uri: str = "sqlite:///heart_disease.db",
) -> optuna.study.Study:
    """
    Performs hyperparameter optimization for LightGBM using Optuna and logs results to MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training labels.
    n_trials : int
        Number of Optuna optimization trials.
    experiment : str
        MLflow experiment name.
    k : int
        Number of cross-validation folds.
    early_stopping_rounds : int, optional (default=10)
        Early stopping parameter for LightGBM.
    default_params : dict, optional
        Default fixed parameters for LightGBM.
    params_to_optimize : dict, optional
        Dictionary of parameter ranges to optimize.
    tracking_uri : str, optional
        MLflow tracking URI.

    Returns
    -------
    optuna.study.Study
        The resulting Optuna study object containing the best trial and optimization history.
    """
    mlflow.set_tracking_uri(tracking_uri)
    current_experiment = mlflow.set_experiment(experiment)

    def objective(
        trial,
        X_train,
        y_train,
        k,
        early_stopping_rounds,
        params_to_optimize,
        default_params,
    ):
        with mlflow.start_run() as run:
            # Definition of hyperparameters to optimize
            parameters = default_params
            for i in params_to_optimize:
                if isinstance(params_to_optimize[i][0], int):
                    parameters[i] = trial.suggest_int(
                        i, params_to_optimize[i][0], params_to_optimize[i][1]
                    )
                elif isinstance(params_to_optimize[i][0], float):
                    log = params_to_optimize[i][0] * 10 < params_to_optimize[i][1]
                    parameters[i] = trial.suggest_float(
                        i, params_to_optimize[i][0], params_to_optimize[i][1], log=log
                    )
            mlflow.log_params({"model": "Boost", "loss": "BCE", **parameters})

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            val_scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(
                    **parameters,
                    n_estimators=1000,
                    n_jobs=-2,
                    class_weight="balanced",
                    verbose=-1,
                )
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="average_precision",
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds, verbose=False)
                    ],
                )

                y_proba_val = model.predict_proba(X_val)[:, 1]
                val_scores.append(average_precision_score(y_val, y_proba_val))
            mlflow.log_metric("average_precision", np.mean(val_scores))

        return np.mean(val_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_ei_candidates=48, multivariate=True),
    )
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            k,
            early_stopping_rounds,
            params_to_optimize,
            default_params,
        ),
        n_trials=n_trials,
    )

    return study


def convert_shap_logit(shap_values_instance: shap.Explanation) -> shap.Explanation:
    """
    Converts SHAP values from log-odds space to probability space for interpretability.

    Parameters
    ----------
    shap_values_instance : shap.Explanation
        SHAP values object generated in logit scale.

    Returns
    -------
    shap.Explanation
        SHAP values converted to probability scale.
    """
    logit_pred = shap_values_instance.base_values + shap_values_instance.values.sum()
    proba_pred = 1 / (1 + np.exp(-logit_pred))
    base_proba = 1 / (1 + np.exp(-shap_values_instance.base_values))
    shap_proba_values = shap_values_instance.values * (proba_pred * (1 - proba_pred))
    shap_values_proba = shap.Explanation(
        values=shap_proba_values,
        base_values=base_proba,
        data=shap_values_instance.data,
        feature_names=shap_values_instance.feature_names,
    )
    return shap_values_proba
