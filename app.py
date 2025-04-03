import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from flask import Flask, request, jsonify, render_template

# Load dataset
df = pd.read_csv("car.csv")
df = df.drop(columns=["Car_Name"])  # Remove unnecessary column

# One-hot encoding categorical variables
df = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True)

# Splitting features and target variable
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Flask app for deployment
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    features = np.array(data).reshape(1, -1)
    
    print("Expected features:", X_train.shape[1])  # Should match input length
    print("Received features:", len(data))

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return render_template("index.html", prediction_text=f"Estimated Price: {prediction:.2f} Lakhs")

if __name__ == '__main__':
    app.run(debug=True)
