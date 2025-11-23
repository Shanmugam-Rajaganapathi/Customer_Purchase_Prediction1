import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, hf_hub_download

# Authenticate Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))

# Download the clean source CSV from HF
csv_path = hf_hub_download(
    repo_id="ShanRaja/Customer-Purchase-Prediction1",
    repo_type="dataset",
    filename="tourism1.csv"
)

# Read CSV and remove any accidental index/Unnamed columns
df = pd.read_csv(csv_path, index_col=False)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop CustomerID column if present
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Fix inconsistent labels in 'Gender'
gender_map = {
    'Male': 'Male',
    'Female': 'Female',
    'Fe Male': 'Female'
}
df['Gender'] = df['Gender'].map(gender_map)

print("Dataset loaded and cleaned successfully.")

# Ensure correct data types
df['CityTier'] = df['CityTier'].astype(int)

# Split features and target
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="ShanRaja/Customer-Purchase-Prediction1",
        repo_type="dataset",
    )

print("All files uploaded to Hugging Face dataset repo successfully!")
