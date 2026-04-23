import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

# Basic overview
print("Shape:", df.shape)

# First few rows
print(df.head())

# Info about dataset
print(df.info())

# Check missing values
print(df.isnull().sum())

# Drop columns with too many missing values (>50%)
missing_percent = df.isnull().mean() * 100

cols_to_drop = missing_percent[missing_percent > 50].index
df.drop(columns=cols_to_drop, inplace=True)

# Fill numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("None")

print("Remaining missing values:\n", df.isnull().sum().sum())

# Convert categorical to numerical
df = pd.get_dummies(df, drop_first=True)

print("New shape:", df.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

from sklearn.ensemble import RandomForestRegressor

# Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train
rf.fit(X_train, y_train)

# Predict
rf_pred = rf.predict(X_test)

# Evaluate
rf_mse = mean_squared_error(y_test, rf_pred)

print("Random Forest MSE:", rf_mse)

from sklearn.metrics import r2_score

# R2 Score
lr_r2 = r2_score(y_test, predictions)
rf_r2 = r2_score(y_test, rf_pred)

print("Linear Regression R2:", lr_r2)
print("Random Forest R2:", rf_r2)