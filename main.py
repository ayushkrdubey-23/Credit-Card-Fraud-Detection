import pandas as pd
import numpy as np

# Visualization
import os
import seaborn as sns
import matplotlib.pyplot as plt
#Training models
import joblib


# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Handle imbalance
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
print("Loading dataset...\n")

df = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded Successfully\n")

# -------------------------------
# 2. BASIC INFO (EDA)
# -------------------------------
print("Dataset Info:")
print("Shape:", df.shape)

print("\nFraud vs Normal:")
print(df["Class"].value_counts())

print("\nPercentage:")
print(df["Class"].value_counts(normalize=True) * 100)

print("\nMissing Values:")
print(df.isnull().sum())
#images/fraud_distribution.png
plt.figure()
df["Class"].value_counts().plot(kind='bar')
plt.title("Fraud vs Normal Transactions")
plt.savefig("images/fraud_distribution.png")
plt.close()
#amount distribution
plt.figure()
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.savefig("images/amount_distribution.png")
plt.close()

# -------------------------------
# 3. SPLIT FEATURES & TARGET
# -------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------
# 4. HANDLE IMBALANCE (SMOTE)
# -------------------------------
print("\n Applying SMOTE to balance data...")

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("\nAfter SMOTE:")
print(y_res.value_counts())

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

print("Data split done")

# -------------------------------
# 6. MODEL TRAINING
# -------------------------------
print("\nTraining model..")
print("Training started... please wait")

model = RandomForestClassifier(
    n_estimators=50,   # fewer trees → faster
    max_depth=10,      # limit tree depth
    random_state=42,
    n_jobs=-1          # use all CPU cores
)
model.fit(X_train, y_train)

print("Model trained successfully")

#Save Models
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")

print("Model saved in models/")

# -------------------------------
# 7. PREDICTION
# -------------------------------
print("\nMaking predictions...")

y_pred = model.predict(X_test)

# -------------------------------
# 8. EVALUATION
# -------------------------------
print("\nModel Evaluation:\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#SAVE REPORT
os.makedirs("outputs", exist_ok=True)
with open("outputs/report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

print("Report saved in outputs/")
# -------------------------------
# 9. VISUALIZATION
# Create folder automatically (BEST PRACTICE)
# -------------------------------
os.makedirs("images", exist_ok=True)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save image
plt.savefig("images/confusion_matrix.png")

print("Image saved successfully!")

plt.show()
plt.close()
print("\nProcess Completed Successfully!")

