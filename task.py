import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

np.random.seed(42)

df = pd.read_csv('processed_spotify_data.csv')

features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'duration_ms']
X = df[features]

# point 2 - regression task (predict singer popularity)
y_reg = df['Artist_popularity']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

models_reg = {
    'Linear regression': LinearRegression(),
    'Decision tree': DecisionTreeRegressor(random_state=42, max_depth=5),
    'Random forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
}

print("Regression results (predicting singer popularity):\n")

for name, model in models_reg.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)
    
    print(f"Model: {name}\n")
    print(f"Mean abs error: {mean_absolute_error(y_test_reg, y_pred):.4f}")
    print(f"Mean sqr error: {mean_squared_error(y_test_reg, y_pred):.4f}")
    print(f"Determination coeff:   {r2_score(y_test_reg, y_pred):.4f}\n")

# point 3 - classification task (predict mode 1-major, 0-minor)
y_clf = df['mode_1.0'].astype(int)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

# scale features for logistic regression
scaler = StandardScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

models_clf = {
    'Logistic regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
    'Decision tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
}

print("\nClassification results (predicting mode_1.0):\n")

plt.figure(figsize=(10, 6))

for name, model in models_clf.items():
    # use scaled data only for logistic regression
    if name == 'Logistic regression':
        model.fit(X_train_clf_scaled, y_train_clf)
        y_pred = model.predict(X_test_clf_scaled)
        y_prob = model.predict_proba(X_test_clf_scaled)[:, 1]
    else:
        model.fit(X_train_clf, y_train_clf)
        y_pred = model.predict(X_test_clf)
        y_prob = model.predict_proba(X_test_clf)[:, 1]
    
    print(f"Model: {name}\n")
    print(f"Accuracy: {accuracy_score(y_test_clf, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_test_clf, y_pred, zero_division=0))
    
    # building ROC-line
    fpr, tpr, _ = roc_curve(y_test_clf, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC-line for classification models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# showing instability of trees

# tree without depth limit
dt_overfit = DecisionTreeRegressor(random_state=42) 
dt_overfit.fit(X_train_reg, y_train_reg)

print(f"\nTrain R2: {r2_score(y_train_reg, dt_overfit.predict(X_train_reg)):.4f}")
print(f"Test R2:   {r2_score(y_test_reg, dt_overfit.predict(X_test_reg)):.4f}")
print("Tree perfectly remembered train but failed on test = unstable")