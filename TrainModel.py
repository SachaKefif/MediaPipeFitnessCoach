import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the recorded data
df_0 = pd.read_csv("dataset_label_0.csv", header=None)
df_1 = pd.read_csv("dataset_label_1.csv", header=None)

# Merge datasets
df_all = pd.concat([df_0, df_1], ignore_index=True)

# 2. Split into features (X) and labels (y)
# The label is the last column
X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train XGBoost
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 5. Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the model to a file so we can load it in real-time later
model.save_model("fitness_coach_xgboost.json")
print("Model saved to fitness_coach_xgboost.json")