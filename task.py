import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    root_mean_squared_error, 
    mean_absolute_error
)

np.random.seed(42)

df_final = pd.read_csv("processed_spotify_data.csv")

print("Regression task solving\n")

X_reg = df_final.drop(columns=['energy'])
X_reg = X_reg.select_dtypes(include=[np.number]) 
y_reg = df_final['energy']

# point 1
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

X_test_reg, X_val_reg, y_test_reg, y_val_reg = train_test_split(
    X_test_reg, y_test_reg, test_size=0.4, random_state=42
)

# scaled for better model results
scaler = StandardScaler()
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)
X_val_reg = scaler.transform(X_val_reg)

# point 2
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

y_pred_reg = lr_model.predict(X_test_reg)

# point 3
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = root_mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"Statistics for linear regression:")
print(f"-mean squared error:  {mse:.4f}")
print(f"-root mean squared error: {rmse:.4f}")
print(f"-mean absolute error:  {mae:.4f}")

ridge_model = Ridge(alpha=1.0, solver='sag', max_iter=2000)
ridge_model.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_reg)
rmse_ridge = root_mean_squared_error(y_test_reg, y_pred_ridge)
print(f"\nRoot mean squared error with ridge (L2) regulation: {rmse_ridge:.4f}\n")
