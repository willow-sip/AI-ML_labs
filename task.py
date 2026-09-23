import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

np.random.seed(42)

df = pd.read_csv('processed_spotify_data.csv')

features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'duration_ms']
X = df[features]
y = df['mode_1.0'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# point 2 - random forest with oob evaluation
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print("Random forest results:")
print(f"OOB score: {rf.oob_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}\n")

# point 3 - adaboost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

print("AdaBoost results:")
print(f"Test accuracy: {accuracy_score(y_test, ada.predict(X_test)):.4f}\n")

# point 3 - gradient boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

print("Gradient boosting results:")
print(f"Test accuracy: {accuracy_score(y_test, gb.predict(X_test)):.4f}\n")

# build roc lines
plt.figure(figsize=(8, 6))

for name, model in [('random forest', rf), ('adaboost', ada), ('gradient boosting', gb)]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (auc = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC lines for ensemble models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()