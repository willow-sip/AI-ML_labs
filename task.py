import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

df_final = pd.read_csv("processed_spotify_data.csv")

print("Regression task solving\n\n")

X_reg = df_final.drop(columns=['energy'])
y_reg = df_final['energy']

# point 1
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

X_test_reg, X_val_reg, y_test_reg, y_val_reg = train_test_split(
    X_test_reg, y_test_reg, test_size=0.4, random_state=42
)