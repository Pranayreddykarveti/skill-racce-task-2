#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Data Collection
# For demonstration, we'll create a synthetic dataset
# You can replace this with actual historical ride request data
data = {
    'date': pd.date_range(start='1/1/2021', periods=500, freq='H'),
    'temperature': np.random.uniform(10, 35, 500),
    'humidity': np.random.uniform(30, 70, 500),
    'is_holiday': np.random.randint(0, 2, 500),
    'ride_requests': np.random.poisson(20, 500)
}
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Convert date to datetime and extract features
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Step 3: Exploratory Data Analysis (EDA)
# Plotting ride requests over time
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['ride_requests'])
plt.xlabel('Date')
plt.ylabel('Ride Requests')
plt.title('Ride Requests Over Time')
plt.show()

# Step 4: Feature Selection
features = ['hour', 'day_of_week', 'month', 'temperature', 'humidity', 'is_holiday']
X = df[features]
y = df['ride_requests']

# Step 5: Model Building
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Step 7: Model Tuning
# (For simplicity, this step is omitted. You can try different algorithms, hyperparameters, etc.)

# Step 8: Prediction
# Predict ride requests for a specific hour (e.g., 10 AM, a random day in January)
sample_input = np.array([[10, 3, 1, 20, 50, 0]])  # hour, day_of_week, month, temperature, humidity, is_holiday
predicted_rides = model.predict(sample_input)
print(f'Predicted Ride Requests for the given hour: {predicted_rides[0]}')


# In[ ]:




