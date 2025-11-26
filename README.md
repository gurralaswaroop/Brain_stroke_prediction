# Brain Stroke Prediction System

A machine learning-powered web application that predicts the risk of brain stroke based on patient data. The system uses a Random Forest Classifier and provides a modern, user-friendly interface for medical professionals.

## Features

- **Accurate Prediction**: Uses a trained Machine Learning model to assess stroke risk.
- **Modern UI**: Clean, glassmorphism-based design for a professional look.
- **Secure Access**: Simple login system for authorized access.
- **Responsive Design**: Works seamlessly on desktop and mobile devices.
- **Instant Results**: Provides immediate risk assessment with probability scores.

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3 (Modern Design System)
- **Machine Learning**: Scikit-learn, Pandas, NumPy

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/gurralaswaroop/Brain_stroke_prediction.git
    cd Brain_stroke_prediction
    ```

2.  **Create a virtual environment** (Optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**:
    ```bash
    python app.py
    ```

5.  **Access the app**:
    Open your browser and go to `http://127.0.0.1:5000/`.

## Usage

1.  **Login**: Use the default credentials:
    *   Username: `admin`
    *   Password: `admin`
2.  **Enter Data**: Fill in the patient's demographic and medical details.
3.  **Predict**: Click "Analyze Risk Profile" to see the result.

## Model Information

The model is trained on a dataset including features like age, hypertension, heart disease, BMI, glucose levels, and smoking status. It uses label encoding for categorical variables and standard scaling for numerical features.
