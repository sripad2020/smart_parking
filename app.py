from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import google.generativeai as genai
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = "your_secret_key"


def convert_paragraph_to_points(paragraph, num_points=5):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_word_tokens = word_tokenize(sentence.lower())
        sentence_word_tokens = [word for word in sentence_word_tokens if word.isalnum()]
        score = sum(freq_dist.get(word, 0) for word in sentence_word_tokens)
        sentence_scores[sentence] = score
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    key_points = sorted_sentences[:min(num_points, len(sorted_sentences))]
    return key_points


def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


# Load and preprocess dataset
data = pd.read_csv('IIoT_Smart_Parking_Management.csv')

# Convert Timestamp to hour
data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.hour

# Encode categorical columns
categorical_columns = data.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Remove outliers
outlier = []
for i in data.select_dtypes(include='number').columns:
    data['scores'] = (data[i] - data[i].mean()) / data[i].std()
    outliers = np.abs(data['scores'] > 3).sum()
    if outliers > 0:
        outlier.append(i)

thresh = 3
for i in outlier:
    upper = data[i].mean() + thresh * data[i].std()
    lower = data[i].mean() - thresh * data[i].std()
    data = data[(data[i] > lower) & (data[i] < upper)]

# Drop temporary scores column and Reserved_Status
data = data.drop(columns=['scores', 'Reserved_Status'])

# Train models for each target
targets = ['Parking_Violation', 'Electric_Vehicle']
models = {}
selected_features_dict = {}

for target in targets:
    corr = data.corr()[target]
    corr = corr.drop([target])
    selected_features = [i for i in corr.index if corr[i] > 0]
    X = data[selected_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    models[target] = model
    selected_features_dict[target] = selected_features
    joblib.dump(model, f'xgboost_model_{target.lower()}.pkl')
    joblib.dump(selected_features, f'selected_features_{target.lower()}.pkl')

# Save encoders
for col, le in label_encoders.items():
    joblib.dump(le, f'encoder_{col}.pkl')

# Load models and encoders
models = {target: joblib.load(f'xgboost_model_{target.lower()}.pkl') for target in targets}
selected_features_dict = {target: joblib.load(f'selected_features_{target.lower()}.pkl') for target in targets}
label_encoders = {col: joblib.load(f'encoder_{col}.pkl') for col in categorical_columns}

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyBvph-JgoPgpF51Fb-0Q-9ikeVwaaCTE2A'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


@app.route('/')
def index():
    return render_template('index.html')


from flask import Flask, render_template, request
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key

# Mock implementations for clean_markdown and convert_paragraph_to_points (replace with your actual functions)
def clean_markdown(text):
    """Remove markdown formatting and clean text."""
    return text.replace('#', '').replace('*', '').strip()

def convert_paragraph_to_points(text, num_points=5):
    """Convert text to a list of bullet points."""
    lines = text.split('\n')
    points = [line.strip('- ').strip() for line in lines if line.strip()]
    return points[:num_points] if len(points) >= num_points else points + [''] * (num_points - len(points))

@app.route('/parking')
def parking_form():
    return render_template('parking.html', csrf_token=app.config.get('WTF_CSRF_SECRET_KEY', 'default-csrf-token'))

@app.route('/parking_submit', methods=['POST'])
def parking_submit():
    try:
        # Validate CSRF token
        if not request.form.get('csrf_token'):
            logging.error("CSRF token missing")
            return render_template('parking.html', data={}, prediction="Error: CSRF token missing",
                                   explanation_points=["Please enable CSRF protection and try again."])

        # Collect form data
        data = {
            'Parking_Spot_ID': int(request.form['Parking_Spot_ID']),
            'Sensor_Reading_Pressure': float(request.form['Sensor_Reading_Pressure']),
            'Vehicle_Type_Weight': float(request.form['Vehicle_Type_Weight']),
            'Vehicle_Type_Height': float(request.form['Vehicle_Type_Height']),
            'User_Type': request.form['User_Type'],
            'Weather_Temperature': float(request.form['Weather_Temperature']),
            'Nearby_Traffic_Level': request.form['Nearby_Traffic_Level'],
            'Entry_Time': int(request.form['Entry_Time']),
            'Payment_Status': request.form['Payment_Status'],
            'Sensor_Reading_Ultrasonic': float(request.form['Sensor_Reading_Ultrasonic']),
            'Parking_Duration': int(request.form['Parking_Duration'])
        }
        logging.debug(f"Form data received: {data}")

        # Validate input ranges
        if data['Entry_Time'] < 0 or data['Entry_Time'] > 23:
            logging.error(f"Invalid Entry Time: {data['Entry_Time']}")
            return render_template('parking.html', data=data, prediction="Error: Invalid Entry Time",
                                   explanation_points=["Entry Time must be between 0 and 23."])
        if data['Parking_Duration'] < 0:
            logging.error(f"Invalid Parking Duration: {data['Parking_Duration']}")
            return render_template('parking.html', data=data, prediction="Error: Invalid Parking Duration",
                                   explanation_points=["Parking Duration cannot be negative."])

        # Prepare data for prediction
        input_df = pd.DataFrame([data])
        for col in ['User_Type', 'Nearby_Traffic_Level', 'Payment_Status']:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].values)
                except ValueError as e:
                    logging.error(f"Label encoding error for {col}: {str(e)}")
                    return render_template('parking.html', data=data,
                                           prediction="Error: Invalid categorical value",
                                           explanation_points=[f"Invalid input for {col}: {str(e)}"])

        for col in selected_features_dict['Parking_Violation']:
            if col not in input_df.columns:
                input_df[col] = 0

        # Make prediction
        features = input_df[selected_features_dict['Parking_Violation']]
        prediction = models['Parking_Violation'].predict(features)[0]
        prediction_label = "Violation" if prediction == 1 else "No Violation"
        logging.debug(f"Prediction: {prediction_label}")

        # Generate explanation using Gemini model
        try:
            prompt = f"""
            Analyze the following parking features and explain why they might indicate a {prediction_label} in a smart parking system:

            - Sensor Reading Pressure: {data['Sensor_Reading_Pressure']}
            - Vehicle Type Weight: {data['Vehicle_Type_Weight']}
            - User Type: {data['User_Type']}
            - Payment Status: {data['Payment_Status']}
            - Parking Duration: {data['Parking_Duration']}

            Provide:
            1. A simple explanation of what a parking violation means in this context.
            2. How these specific feature values suggest this prediction.
            3. Recommended mitigation steps to avoid violations.
            4. Potential false positives to consider (e.g., sensor errors).

            Format the response in clear bullet points.
            """
            response = gemini_model.generate_content(prompt)
            explanation = clean_markdown(response.text)
            explanation_points = convert_paragraph_to_points(explanation, num_points=5)
            logging.debug(f"Gemini explanation: {explanation_points}")
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            explanation_points = [f"Could not generate explanation: {str(e)}"]

        return render_template('parking.html', data=data, prediction=prediction_label,
                               explanation_points=explanation_points)

    except Exception as e:
        logging.error(f"Unexpected error in parking_submit: {str(e)}")
        return render_template('parking.html', data={}, prediction="Error: Submission failed",
                               explanation_points=[f"An unexpected error occurred: {str(e)}"])
@app.route('/electric')
def electric_form():
    return render_template('electric_vehicle.html')


@app.route('/electric_submit', methods=['POST'])
def electric_submit():
    data = {
        "Parking_Spot_ID": int(request.form['Parking_Spot_ID']),
        "Sensor_Reading_Proximity": float(request.form['Sensor_Reading_Proximity']),
        "Vehicle_Type_Weight": float(request.form['Vehicle_Type_Weight']),
        "Exit_Time": int(request.form['Exit_Time']),
        "Occupancy_Rate": float(request.form['Occupancy_Rate']),
        "Parking_Lot_Section": request.form['Parking_Lot_Section'],
        "Parking_Duration": int(request.form['Parking_Duration']),
        "Environmental_Noise_Level": float(request.form['Environmental_Noise_Level']),
        "Proximity_To_Exit": float(request.form['Proximity_To_Exit']),
        "User_Parking_History": float(request.form['User_Parking_History']),
    }

    input_df = pd.DataFrame([data])
    for col in ['Parking_Lot_Section']:
        if col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].values)
            except ValueError as e:
                return render_template('electric_vehicle.html', data=data,
                                       prediction="Error: Invalid categorical value",
                                       explanation_points=[f"Invalid input for {col}: {str(e)}"])

    for col in selected_features_dict['Electric_Vehicle']:
        if col not in input_df.columns:
            input_df[col] = 0

    # Debugging: Check dtypes before prediction
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            return render_template('electric_vehicle.html', data=data,
                                   prediction="Error: Non-numeric column detected",
                                   explanation_points=[f"Column {col} is still of type object"])

    features = input_df[selected_features_dict['Electric_Vehicle']]
    prediction = models['Electric_Vehicle'].predict(features)[0]
    prediction_label = "Electric" if prediction == 1 else "Non-Electric"

    try:
        prompt = f"""
        Analyze the following parking features and explain why they might indicate a {prediction_label} vehicle in a smart parking system:

        - Sensor Reading Proximity: {data['Sensor_Reading_Proximity']}
        - Vehicle Type Weight: {data['Vehicle_Type_Weight']}
        - Parking Lot Section: {data['Parking_Lot_Section']}
        - Occupancy Rate: {data['Occupancy_Rate']}
        - User Parking History: {data['User_Parking_History']}

        Provide:
        1. A simple explanation of what an electric vehicle means in this context.
        2. How these specific feature values suggest this prediction.
        3. Recommended steps for accommodating electric vehicles (e.g., charging stations).
        4. Potential false positives to consider (e.g., weight misclassification).

        Format the response in clear bullet points.
        """
        response = gemini_model.generate_content(prompt)
        explanation = clean_markdown(response.text)
        explanation_points = convert_paragraph_to_points(explanation, num_points=5)
    except Exception as e:
        explanation_points = [f"Could not generate explanation: {str(e)}"]

    return render_template('electric_vehicle.html', data=data, prediction=prediction_label,
                           explanation_points=explanation_points)


if __name__ == '__main__':
    app.run(debug=True)