from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "dataset" / "creditcard.csv"
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError("Place creditcard.csv in dataset/ before running this script.")

df = pd.read_csv(CSV)
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:", int(df.isna().sum().sum()))
print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nFraud percentage:", round(df["Class"].mean() * 100, 4), "%")

plt.figure(figsize=(7,4))
df["Class"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0,1], ["Legitimate", "Fraud"], rotation=0)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(OUT / "real_dataset_class_distribution.png", dpi=160)
plt.close()

plt.figure(figsize=(7,4))
plt.hist(df.loc[df["Class"] == 0, "Amount"].clip(upper=df["Amount"].quantile(.99)),
         bins=50, alpha=.7, label="Legitimate")
plt.hist(df.loc[df["Class"] == 1, "Amount"].clip(upper=df["Amount"].quantile(.99)),
         bins=50, alpha=.7, label="Fraud")
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "real_dataset_amount_distribution.png", dpi=160)
plt.close()
