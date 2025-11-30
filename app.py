from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User
import os
from dotenv import load_dotenv

import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'supersecretkey')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///db.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/brain_stroke.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Brain Stroke Prediction startup')

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load model and preprocessors
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Database tables will be created via init_db.py or manual command


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
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
