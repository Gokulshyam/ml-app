# ml_project.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib   # for saving model

# Step 1: Create simple data
data = {
    'size': [850, 900, 1200, 1500, 1800],
    'bedrooms': [2, 2, 3, 3, 4],
    'price': [100000, 120000, 140000, 170000, 200000]
}
df = pd.DataFrame(data)

# Step 2: Features & Target
X = df[['size', 'bedrooms']]
y = df['price']

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict & Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 6: Save Model
joblib.dump(model, 'house_price_model.pkl')
print("✅ Model saved as house_price_model.pkl")

