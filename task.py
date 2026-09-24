import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
X = np.random.randint(0, 2, size=(200, 12))


true_y = (X[:, :6].sum(axis=1) > 3).astype(int)
noise = np.random.random(len(X)) < 0.1
y = true_y.copy()
y[noise] = 1 - y[noise]

Y_onehot = np.zeros((len(y), 2))
Y_onehot[np.arange(len(y)), y] = 1

np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y_onehot, fmt='%d')

print(f"class distribution: class 0 = {len(y) - y.sum()}, class 1 = {y.sum()}")
print(f"total samples: {len(y)}\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# build mlp
model = Sequential([
    Dense(32, activation='sigmoid', input_shape=(12,)),
    Dense(16, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=200, 
    batch_size=16,
    validation_data=(X_test, y_test), 
    callbacks=[early_stop],
    verbose=0
)

y_pred_mlp = (model.predict(X_test) > 0.5).astype("int32").flatten()
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

print("Mlp results:")
print(f"Accuracy: {mlp_accuracy:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred_mlp, zero_division=0))

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Mlp training history')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_mlp)
plt.imshow(cm, cmap='Blues')
plt.title('Mlp confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.show()

print("\nComparison with classical models:")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"Logistic regression accuracy: {accuracy_score(y_test, lr_pred):.4f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random forest accuracy: {accuracy_score(y_test, rf_pred):.4f}")

print(f"\nBest model: {'mlp' if mlp_accuracy >= max(accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred)) else 'classical'}")