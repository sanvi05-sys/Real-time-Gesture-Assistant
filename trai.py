import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

# Load processed data
X = np.load('data/processed/X.npy')  # Hand landmark features
y = np.load('data/processed/y.npy')  # Corresponding labels
gesture_names = np.load('data/processed/gesture_names.npy')  # Gesture class names

print(f"Loaded {len(X)} samples with {len(gesture_names)} gesture classes")
print("Gesture classes:", list(gesture_names))

# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class balance
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Model Architecture
model = Sequential([
    # Input layer (63 features: 21 landmarks * 3 coordinates)
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),  # Normalizes activations
    Dropout(0.3),  # Randomly disable 30% neurons to prevent overfitting
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Output layer with softmax for multi-class classification
    Dense(len(gesture_names), activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(
        'models/best_gesture_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True,
        mode='max'
    )
]

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('models/gesture_model.h5')
print("\nModel saved to 'models/gesture_model.h5'")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_history.png')
print("Training plots saved to 'models/training_history.png'")