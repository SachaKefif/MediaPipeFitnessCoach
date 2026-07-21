import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the recorded data
df_0 = pd.read_csv("dataset_label_0.csv", header=None)
df_1 = pd.read_csv("dataset_label_1.csv", header=None)
df_2 = pd.read_csv("dataset_label_2.csv", header=None)
df_3 = pd.read_csv("dataset_label_3.csv", header=None)

df_all = pd.concat([df_0, df_1, df_2, df_3], ignore_index=True)
print(f"The complete file contains {len(df_all)} rows")

# 2. Split into features (X) and labels (y)
X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values

# 3. Reshape Data for LSTM: [samples, time_steps, features_per_frame]
# A Neural Network requires the data structured in 3D time-blocks, not flattened.
TIME_STEPS = 60
FEATURES_PER_FRAME = X.shape[1] // TIME_STEPS

X = X.reshape(-1, TIME_STEPS, FEATURES_PER_FRAME)

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1415, stratify=y
)

# 5. Initialize the Neural Network (LSTM)
print("Training Neural Network...")
model = Sequential([
    # LSTM layer to process the 60 frames sequentially
    LSTM(64, input_shape=(TIME_STEPS, FEATURES_PER_FRAME), return_sequences=False),

    # Dropout prevents the network from memorizing the data (overfitting)
    Dropout(0.2),

    # Standard hidden layer
    Dense(32, activation='relu'),

    # Output layer: 4 neurons for 4 classes, using softmax for percentages
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Perfect for integer labels (0, 1, 2, 3)
    metrics=['accuracy']
)

# 6. Train the model
class_weights = {
    0: 2.5,  # Makes missing Class 0 hurt more in the loss function, because we want less FN (missed "0")
    1: 1.0,
    2: 1.0,
    3: 1.0
}

history = model.fit(
    X_train, y_train,
    epochs=500,  # Number of full passes over the data
    batch_size=32,  # How many samples processed at once
    class_weight=class_weights, # penality
    validation_split=0.2,  # Uses 20% of training data to validate per epoch
    verbose=1
)

# 6bis. Plot and save accuracy evolution graph
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Evolution Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
# Save image file
plot_filename = "accuracy_evolution.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nAccuracy plot successfully saved as '{plot_filename}'")

# 7. Test metrics
print("\nEvaluating model...")
# predict() gives probabilities; argmax grabs the index (0,1,2,3) with the highest probability
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))
print("Macro Precision:", precision_score(y_test, y_pred, average='macro'))
print("Macro Recall:", recall_score(y_test, y_pred, average='macro'))

# 8. Save the Neural Network model
model.save("fitness_coach_lstm.keras")
print("Model saved to fitness_coach_lstm.keras")