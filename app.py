from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sqlite3
import secrets
from datetime import datetime
from groq import Groq
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configure Groq client
groq_api_key = "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"
groq_client = Groq(api_key=groq_api_key)

# Load the trained ML model
model_path = os.path.join(os.getcwd(), 'my_modal.pkl')
model = joblib.load(model_path)
from tensorflow.keras.models import load_model
import numpy as np

# Load TensorFlow NN model and scaler
nn_model = load_model('model/nn_model.h5')
nn_scaler = joblib.load('model/scaler.pkl')
# Database setup
def get_db_connection():
    conn = sqlite3.connect('mental_health.db', timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        result TEXT NOT NULL,
        recommendations TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# ML model columns
trained_columns = [
    'id', 'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
    'Sleep Duration', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
    'Family History of Mental Illness', 'City_Agra', 'City_Ahmedabad', 'City_Bangalore', 'City_Bhavna', 'City_Bhopal', 
    'City_Chennai', 'City_City', 'City_Delhi', 'City_Faridabad', 'City_Gaurav', 'City_Ghaziabad', 'City_Harsh', 
    'City_Harsha', 'City_Hyderabad', 'City_Indore', 'City_Jaipur', 'City_Kalyan', 'City_Kanpur', 'City_Khaziabad', 
    'City_Kibara', 'City_Kolkata', 'City_Less Delhi', 'City_Less than 5 Kalyan', 'City_Lucknow', 'City_Ludhiana', 
    'City_M.Com', 'City_M.Tech', 'City_ME', 'City_Meerut', 'City_Mihir', 'City_Mira', 'City_Mumbai', 'City_Nagpur', 
    'City_Nalini', 'City_Nalyan', 'City_Nandini', 'City_Nashik', 'City_Patna', 'City_Pune', 'City_Rajkot', 'City_Rashi', 
    'City_Reyansh', 'City_Saanvi', 'City_Srinagar', 'City_Surat', 'City_Thane', 'City_Vaanya', 'City_Vadodara', 
    'City_Varanasi', 'City_Vasai-Virar', 'City_Visakhapatnam', 'Profession_Chef', 'Profession_Civil Engineer', 
    'Profession_Content Writer', 'Profession_Digital Marketer', 'Profession_Doctor', 'Profession_Educational Consultant', 
    'Profession_Entrepreneur', 'Profession_Lawyer', 'Profession_Manager', 'Profession_Pharmacist', 'Profession_Student', 
    'Profession_Teacher', 'Profession_UX/UI Designer', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 
    'Dietary Habits_Unhealthy', 'Degree_B.Com', 'Degree_B.Ed', 'Degree_B.Pharm', 'Degree_B.Tech', 'Degree_BA', 'Degree_BBA', 
    'Degree_BCA', 'Degree_BE', 'Degree_BHM', 'Degree_BSc', 'Degree_Class 12', 'Degree_LLB', 'Degree_LLM', 'Degree_M.Com', 
    'Degree_M.Ed', 'Degree_M.Pharm', 'Degree_M.Tech', 'Degree_MA', 'Degree_MBA', 'Degree_MBBS', 'Degree_MCA', 'Degree_MD', 
    'Degree_ME', 'Degree_MHM', 'Degree_MSc', 'Degree_Others', 'Degree_PhD'
]

# Function to preprocess the input data
def preprocess_data(input_data):
    # Encoding categorical features
    sleep_duration_map = {
        'Less than 5 hours': 1,
        '5-6 hours': 2,
        '7-8 hours': 3,
        'More than 8 hours': 4
    }

    # Dictionary for encoding binary categorical variables
    label_encoder_dict = {
        'Gender': {'Male': 1, 'Female': 0},
        'Have you ever had suicidal thoughts ?': {'Yes': 1, 'No': 0},
        'Family History of Mental Illness': {'Yes': 1, 'No': 0}
    }

    # Encoding categorical variables
    input_data['Gender'] = label_encoder_dict['Gender'][input_data['Gender'][0]]
    input_data['Have you ever had suicidal thoughts ?'] = label_encoder_dict['Have you ever had suicidal thoughts ?'][input_data['Have you ever had suicidal thoughts ?'][0]]
    input_data['Family History of Mental Illness'] = label_encoder_dict['Family History of Mental Illness'][input_data['Family History of Mental Illness'][0]]
    input_data['Sleep Duration'] = sleep_duration_map[input_data['Sleep Duration'][0]]

    # One-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data, columns=['City', 'Profession', 'Dietary Habits', 'Degree'], drop_first=True)

    # Scaling the numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
                          'Work/Study Hours', 'Financial Stress']
    input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

    # Ensure that the input data has the same columns as the training data
    input_data = input_data.reindex(columns=trained_columns, fill_value=0)

    return input_data

# Function to make a prediction
def make_prediction(input_data):
    # Preprocess the input data
    processed_data = preprocess_data(input_data)
    
    # Predict using the trained model
    prediction = model.predict(processed_data)
    
    # Return the prediction result
    return prediction

# Get AI recommendations using Groq
def get_groq_recommendations(user_data, prediction_result):
    try:
        prompt = f"""
        I need advice for improving mental health for a {user_data['age']}-year-old {user_data['gender']} 
        who is studying {user_data['degree']} and works as a {user_data['profession']}.
        
        Mental health assessment data:
        - Academic Pressure: {user_data['academic_pressure']}/10
        - Work Pressure: {user_data['work_pressure']}/10
        - CGPA: {user_data['cgpa']}/10
        - Sleep Duration: {user_data['sleep_duration']}
        - Has experienced suicidal thoughts: {user_data['suicidal_thoughts']}
        - Financial Stress Level: {user_data['financial_stress']}/5
        - Family History of Mental Illness: {user_data['family_history']}
        
        ML model prediction: {prediction_result}
        
        Please provide:
        1. 🔍 A brief analysis of their mental health situation
        2. 🌱 Five personalized recommendations to improve their mental health
        3. 💪 A motivational quote relevant to their situation
        4. 🎬 Suggest two types of uplifting content they could watch or read
        
        Use emojis and format your response in a friendly, supportive way.
        """
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return "We're experiencing issues with our recommendation system. Please try again later."

# Get a random motivational quote
def get_random_quote():
    quotes = [
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "It does not matter how slowly you go as long as you do not stop. - Confucius",
        "Everything you've ever wanted is on the other side of fear. - George Addair"
    ]
    return secrets.choice(quotes)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']

        with get_db_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

            if user:
                flash('Username already exists. Please choose a different one.', 'danger')
                return redirect(url_for('register'))

            conn.execute('INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)',
                         (username, password, name, email))
            conn.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
                            (username, password)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['name'] = user['name']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access your dashboard.', 'warning')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    assessments = conn.execute('SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC', 
                              (session['user_id'],)).fetchall()
    conn.close()
    
    quote = get_random_quote()
    
    return render_template('dashboard.html', assessments=assessments, quote=quote)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to access the assessment.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get user inputs from form
        gender = request.form.get('gender')
        age = float(request.form.get('age'))
        academic_pressure = float(request.form.get('academic_pressure'))
        work_pressure = float(request.form.get('work_pressure'))
        cgpa = float(request.form.get('cgpa'))
        study_satisfaction = float(request.form.get('study_satisfaction'))
        job_satisfaction = float(request.form.get('job_satisfaction'))
        sleep_duration = request.form.get('sleep_duration')
        suicidal_thoughts = request.form.get('suicidal_thoughts')
        work_study_hours = float(request.form.get('work_study_hours'))
        financial_stress = float(request.form.get('financial_stress'))
        family_history = request.form.get('family_history')
        city = request.form.get('city')
        profession = request.form.get('profession')
        dietary_habits = request.form.get('dietary_habits')
        degree = request.form.get('degree')
        
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Academic Pressure': [academic_pressure],
            'Work Pressure': [work_pressure],
            'CGPA': [cgpa],
            'Study Satisfaction': [study_satisfaction],
            'Job Satisfaction': [job_satisfaction],
            'Sleep Duration': [sleep_duration],
            'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
            'Work/Study Hours': [work_study_hours],
            'Financial Stress': [financial_stress],
            'Family History of Mental Illness': [family_history],
            'City': [city],
            'Profession': [profession],
            'Dietary Habits': [dietary_habits],
            'Degree': [degree]
        })
        
        # Make prediction
        prediction = make_prediction(input_data)
        result = 'Depressed' if prediction[0] == 1 else 'Not Depressed'
        
        # Get AI recommendations
        user_data = {
            'gender': gender,
            'age': age,
            'academic_pressure': academic_pressure,
            'work_pressure': work_pressure,
            'cgpa': cgpa,
            'sleep_duration': sleep_duration,
            'suicidal_thoughts': suicidal_thoughts,
            'financial_stress': financial_stress,
            'family_history': family_history,
            'profession': profession,
            'degree': degree
        }
        
        recommendations = get_groq_recommendations(user_data, result)
        
        # Save assessment to database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO assessments (user_id, result, recommendations)
                    VALUES (?, ?, ?)
                    ''', (session['user_id'], result, recommendations))
            conn.commit()

        
        return render_template('result.html', result=result, recommendations=recommendations)
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)