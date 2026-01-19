import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

print("Unique PCOS (Y/N) values before mapping:", df['PCOS (Y/N)'].unique())
df['PCOS'] = df['PCOS (Y/N)'].map({'Y': 1, 'N': 0})
print("Rows before dropping NaN PCOS:", len(df))
df = df.dropna(subset=['PCOS'])
df['PCOS'] = df['PCOS'].astype(int)
print("Rows after dropping NaN PCOS:", len(df))

cat_cols = [
    "Blood Group", "Cycle Regularity (Y/N)", "Pregnant (Y/N)", "Hair Loss (Y/N)", "Pimples/Acne (Y/N)",
    "Excess Hair Growth (Y/N)", "Difficulty in Weight Loss (Y/N)", "Skin Darkening (Y/N)", "Smoking (Y/N)",
    "Family History of Hormonal Issues (Y/N)", "Skin Type", "Birth Control (Y/N)", "Regular Medication (Y/N)"
]
num_cols = [
    "Age", "Height(cm)", "Weight(kg)", "BMI", "Cycle Length(days)", "Marriage Status (Yrs)",
    "No. of Abortions", "Exercise (days/week)", "Alcohol (drinks/week)", "Sleep (hours/day)",
    "Stress Level (1-10)", "Water Intake (liters/day)", "Fast Food (meals/week)", "Coffee/Tea (cups/day)"
]

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

features = num_cols + cat_cols
X = df[features]
y = df["PCOS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(features, "models/feature_names.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Saved model, encoders, feature_names, scaler to /models.")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.barh(np.array(features)[indices][:15][::-1], importances[indices][:15][::-1], color="orchid")
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig("models/global_feature_importances.png", dpi=150)
plt.show()

input_row = X_test.iloc[0]
mean_row = pd.Series(X_train.mean(), index=features)
diff = (input_row - mean_row).abs().sort_values(ascending=False)
top_k = 7
plt.figure(figsize=(8,4))
plt.barh(diff.index[:top_k][::-1], diff.values[:top_k][::-1], color="seagreen")
plt.xlabel("Absolute deviation from average user")
plt.title("User Most Distinctive Features (example)")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()