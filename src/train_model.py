import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Step 1: Loading Dataset")

df = pd.read_csv("data/creditcard.csv")

df = df.sample(
    n=50000,
    random_state=42
)

print("Dataset Loaded:", df.shape)

print("Step 2: Scaling Amount")

scaler = StandardScaler()

df['Scaled_Amount'] = scaler.fit_transform(df[['Amount']])

df.drop('Amount', axis=1, inplace=True)

print("Step 3: Splitting Data")

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Step 4: Training Model")

rf = RandomForestClassifier(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Step 5: Saving Model")
joblib.dump(rf, "fraud_model.pkl")
print("Model Saved Successfully!")