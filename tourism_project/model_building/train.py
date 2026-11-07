# for data manipulation
import os
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

# The workflow starts an MLflow server on localhost:5000 before this script runs
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# Set the class weight to handle class imbalance (~19% buyers)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],       # number of boosting trees
    "xgbclassifier__max_depth": [3, 5],              # tree depth (complexity)
    "xgbclassifier__colsample_bytree": [0.6, 0.8],   # feature fraction per tree
    "xgbclassifier__colsample_bylevel": [0.6, 0.8],  # feature fraction per level
    "xgbclassifier__learning_rate": [0.05, 0.1],     # shrinkage / step size
    "xgbclassifier__reg_lambda": [1, 5],             # L2 regularisation
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV (F1 suits the imbalanced target)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)
    print("Best params:", grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Classification threshold (kept identical in app.py)
    classification_threshold = 0.45
    mlflow.log_param("classification_threshold", classification_threshold)

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print("Test classification report:")
    print(classification_report(ytest, y_pred_test))

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
