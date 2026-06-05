import pandas as pd
df = pd.read_csv("data/creditcard.csv")
df.to_csv(
    "fraud_analysis.csv",
    index=False
)
print("CSV Exported Successfully")