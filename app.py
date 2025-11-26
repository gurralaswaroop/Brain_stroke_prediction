from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this for production

# Load model and preprocessors
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('predict'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Dummy authentication
    if username == 'admin' and password == 'admin':
        session['user'] = username
        return redirect(url_for('predict'))
    else:
        flash('Invalid Credentials. Try admin/admin', 'error')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Extract data from form
            data = {
                'age': float(request.form['age']),
                'gender': request.form['gender'],
                'marital_family_history': request.form['marital_family_history'],
                'hypertension': int(request.form['hypertension']),
                'heart_disease': int(request.form['heart_disease']),
                'diabetes_flag': int(request.form['diabetes_flag']),
                'high_cholesterol_flag': int(request.form['high_cholesterol_flag']),
                'previous_stroke_tia': int(request.form['previous_stroke_tia']),
                'kidney_disease': int(request.form['kidney_disease']),
                'bmi': float(request.form['bmi']),
                'systolic_bp': float(request.form['systolic_bp']),
                'diastolic_bp': float(request.form['diastolic_bp']),
                'blood_sugar': float(request.form['blood_sugar']),
                'chol_total': float(request.form['chol_total']),
                'chol_hdl': float(request.form['chol_hdl']),
                'chol_ldl': float(request.form['chol_ldl']),
                'chol_triglycerides': float(request.form['chol_triglycerides']),
                'smoking_status': request.form['smoking_status'],
                'alcohol_consumption': request.form['alcohol_consumption'],
                'physical_activity': request.form['physical_activity'],
                'dietary_habits': request.form['dietary_habits'],
                'sleep_hours': float(request.form['sleep_hours'])
            }
            
            # Create DataFrame for processing
            input_df = pd.DataFrame([data])
            
            # Encode categorical variables
            categorical_cols = ['gender', 'marital_family_history', 'smoking_status', 'alcohol_consumption', 'physical_activity', 'dietary_habits']
            for col in categorical_cols:
                if col in label_encoders:
                    # Handle unseen labels if necessary, for now assume valid input
                    input_df[col] = label_encoders[col].transform(input_df[col])
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            return render_template('result.html', prediction=prediction, probability=probability)
            
        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
