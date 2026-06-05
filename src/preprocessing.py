import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("data/creditcard.csv")

# Scale Amount Column
scaler = StandardScaler()

df['Scaled_Amount'] = scaler.fit_transform(
    df[['Amount']]
)

# Drop Original Amount
df.drop(
    'Amount',
    axis=1,
    inplace=True
)

# Features
X = df.drop(
    'Class',
    axis=1
)

# Target
y = df['Class']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Preprocessing Completed")

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)