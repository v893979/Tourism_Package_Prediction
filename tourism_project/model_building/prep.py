import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Remove columns that carry no predictive information:
#   - 'Unnamed: 0' is a leftover row index from the CSV export
#   - 'CustomerID' is a unique identifier
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Basic cleaning: fix the inconsistent 'Fe Male' label and remove duplicate rows
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df = df.drop_duplicates().reset_index(drop=True)

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values, so training and serving use the same representation.

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
