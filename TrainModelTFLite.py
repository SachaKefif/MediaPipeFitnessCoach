import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# 1. Load the recorded data
df_0 = pd.read_csv("dataset_label_0.csv", header=None)
df_1 = pd.read_csv("dataset_label_1.csv", header=None)

# Merge datasets
df_all = pd.concat([df_0, df_1], ignore_index=True)

# 2. Split into features (X) and labels (y)
# The label is the last column
X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values

# 3. Split into training, validation, and testing sets
# 70% training
# 15% validation  <- used during training
# 15% testing    <- final evaluation
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=145
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=5
)

# On normalise les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Initialize and train Model
print("Training Model...")

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 5. Test metrics
y_prob = model.predict(X_test).flatten()
threshold = 0.39   # make it harder to predict 1 because we want no false positive
y_pred = (y_prob >= threshold).astype(int)

# from sklearn.model_selection import cross_val_score
#
# scores = cross_val_score(
#     model,
#     X,
#     y,
#     cv=5,
#     scoring="precision"
# )
#
# print(scores)
# print("Average precision:", scores.mean())

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100}%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("[TN FP]"
      "[FN TP]")
print(cm)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)



# 6. Save the model to a file so we can load it in real-time later
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("fitness_coach.tflite", "wb") as f:
    f.write(tflite_model)

joblib.dump(scaler, "scaler.pkl")