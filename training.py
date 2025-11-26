import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
df = pd.read_csv('sample_stroke_prediction.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Check which columns actually exist in the dataframe
existing_cols = df.columns.tolist()
print(f"Columns in dataset: {existing_cols}")

# Adjust categorical columns based on actual data
# Based on previous `head` output: gender, marital_family_history (maybe?), smoking_status, alcohol_consumption, physical_activity, dietary_habits
# Let's map the standard dataset columns to what we saw or just use what is there.
# The previous head output showed:
# id,age,gender,marital_family_history,hypertension,heart_disease,diabetes_flag,high_cholesterol_flag,previous_stroke_tia,kidney_disease,bmi,systolic_bp,diastolic_bp,blood_sugar,chol_total,chol_hdl,chol_ldl,chol_triglycerides,smoking_status,alcohol_consumption,physical_activity,dietary_habits,sleep_hours,stroke

categorical_cols = ['gender', 'marital_family_history', 'smoking_status', 'alcohol_consumption', 'physical_activity', 'dietary_habits']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Select features and target
# Dropping ID as it's not predictive
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and preprocessors saved.")
