import pandas as pd
import os
import joblib

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score

import xgboost as xgb
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Download preprocessed CSVs from HF
dataset_repo = "ShanRaja/Customer-Purchase-Prediction1"

def download_csv(filename):
    path = hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename=filename)
    return pd.read_csv(path)

# Load train/test
X_train = download_csv("X_train.csv")
X_test  = download_csv("X_test.csv")
y_train = download_csv("y_train.csv").values.ravel()
y_test  = download_csv("y_test.csv").values.ravel()

# Column definitions
binary_cols = ['Gender', 'Passport', 'OwnCar']
onehot_cols = ['TypeofContact','Occupation','MaritalStatus','ProductPitched','Designation']
numeric_cols = [
    'Age', 'CityTier', 'MonthlyIncome', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'NumberOfChildrenVisiting', 'PitchSatisfactionScore',
    'NumberOfFollowups'
]

# Column transformer: one-hot for categorical, ordinal for binary, passthrough for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('binary', OrdinalEncoder(), binary_cols),
        ('numeric', 'passthrough', numeric_cols)
    ]
)

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', xgb_model)
])

# Hyperparameter grid
param_grid = {
    'xgb__n_estimators': [50, 75, 100],
    'xgb__max_depth': [2, 3, 4],
    'xgb__colsample_bytree': [0.4, 0.5, 0.6],
    'xgb__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__reg_lambda': [0.4, 0.5, 0.6],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

# MLflow experiment
mlflow.set_experiment("customer_purchase_prediction1")

with mlflow.start_run():
    # Train
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Log recall metrics
    mlflow.log_metric("train_recall", recall_score(y_train, best_model.predict(X_train)))

    # Print classification reports
    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)

    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_pred_train))

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))

    # Log model in MLflow
    mlflow.sklearn.log_model(best_model, "xgb_pipeline_model")

    # Save locally for HF upload
    model_filename = "best_customer_purchase_prediction_model_v2_23.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Model saved as {model_filename}")

# Upload to Hugging Face
repo_id = "ShanRaja/Customer-Purchase-Prediction"
model_repo_type = "model"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=model_repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=model_repo_type, private=False)
    print(f"Model repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=repo_id,
    repo_type=model_repo_type
)

print("Model uploaded successfully!")
