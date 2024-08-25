from flask import Flask, request, redirect
from pycaret.anomaly import load_model as load_anomaly_model
from pycaret.regression import load_model as load_regression_model
from pycaret.classification import load_model as load_classification_model
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
regression_model = load_regression_model('models/regression_model')
anomaly_model = load_anomaly_model('models/anomaly_detection_model')
classification_model = load_classification_model('models/mushroom_classification_model')

# Define the home route for the HDB Resale Price Prediction App
@app.route('/hdb')
def hdb_home():
    return redirect('https://combined-1-z6te.onrender.com')

# Define the prediction route for HDB Resale Price Prediction App
@app.route('/hdb/predict', methods=['POST'])
def hdb_predict():
    # Get form data
    floor_area = request.form['floor_area']
    cbd_dist = request.form['cbd_dist']
    min_dist_mrt = request.form['min_dist_mrt']

    # Create a DataFrame for prediction with all required columns
    input_data = pd.DataFrame({
        'floor_area_sqm': [float(floor_area)],
        'cbd_dist': [float(cbd_dist)],
        'min_dist_mrt': [float(min_dist_mrt)],
        'block': [None],  # Placeholder
        'street_name': [None],  # Placeholder
        'town': [None],  # Placeholder
        'postal_code': [None],  # Placeholder
        'month': [None],  # Placeholder
        'flat_type': [None],  # Placeholder
        'storey_range': [None],  # Placeholder
        'flat_model': [None],  # Placeholder
        'lease_commence_date': [None],  # Placeholder
        'latitude': [None],  # Placeholder
        'longitude': [None],  # Placeholder
        'flat_age': [None]  # Placeholder
    })

    # Make prediction
    predictions = predict_model(regression_model, data=input_data)

    # Extract the prediction value
    result = predictions['prediction_label'][0]

    # Redirect to the external site with prediction
    return redirect(f'https://combined-1-z6te.onrender.com?prediction={result}')

# Define the home route for fillbird
@app.route('/')
def home():
    return redirect('https://fillbird.onrender.com/')

# Define the prediction route for fillbird
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    input_data = {
        'FISCAL_YR': request.form.get('fiscal_year'),
        'FISCAL_MTH': request.form.get('fiscal_month'),
        'DEPT_NAME': request.form.get('dept_name'),
        'DIV_NAME': request.form.get('div_name'),
        'CAT_DESC': request.form.get('cat_desc'),
        'AMT': request.form.get('amt')
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the AMT column is in numeric format, replacing errors with NaN
    input_df['AMT'] = pd.to_numeric(input_df['AMT'], errors='coerce')

    # Generate predictions
    predictions = predict_model(anomaly_model, data=input_df)
    anomaly_flag = predictions['Anomaly_Score'][0]

    # Convert numerical prediction to "Anomaly" or "Not Anomaly"
    prediction = "Anomaly" if anomaly_flag > 0 else "Not Anomaly"

    # Redirect to the external site with prediction
    return redirect(f'https://fillbird.onrender.com?prediction={prediction}')



# Define the home route for the mushroom classification model
@app.route('/mushroom')
def mushroom_home():
    return redirect('https://mushrooms-1.onrender.com')

# Define the prediction route for the mushroom classification
@app.route('/mushroom/predict', methods=['POST'])
def mushroom_predict():
    # Collect input data from the form
    cap_shape = request.form.get('cap_shape')
    cap_surface = request.form.get('cap_surface')
    cap_color = request.form.get('cap_color')
    bruises = request.form.get('bruises')
    odor = request.form.get('odor')
    gill_attachment = request.form.get('gill_attachment')
    gill_spacing = request.form.get('gill_spacing')
    gill_size = request.form.get('gill_size')
    gill_color = request.form.get('gill_color')
    stalk_shape = request.form.get('stalk_shape')
    stalk_root = request.form.get('stalk_root')
    stalk_surface_above_ring = request.form.get('stalk_surface_above_ring')
    stalk_surface_below_ring = request.form.get('stalk_surface_below_ring')
    stalk_color_above_ring = request.form.get('stalk_color_above_ring')
    stalk_color_below_ring = request.form.get('stalk_color_below_ring')
    veil_type = request.form.get('veil_type')
    veil_color = request.form.get('veil_color')
    ring_number = request.form.get('ring_number')
    ring_type = request.form.get('ring_type')
    spore_print_color = request.form.get('spore_print_color')
    population = request.form.get('population')
    habitat = request.form.get('habitat')

    # Convert the data into a DataFrame
    input_data = pd.DataFrame({
        'cap_shape': [cap_shape],
        'cap_surface': [cap_surface],
        'cap_color': [cap_color],
        'bruises': [bruises],
        'odor': [odor],
        'gill_attachment': [gill_attachment],
        'gill_spacing': [gill_spacing],
        'gill_size': [gill_size],
        'gill_color': [gill_color],
        'stalk_shape': [stalk_shape],
        'stalk_root': [stalk_root],
        'stalk_surface_above_ring': [stalk_surface_above_ring],
        'stalk_surface_below_ring': [stalk_surface_below_ring],
        'stalk_color_above_ring': [stalk_color_above_ring],
        'stalk_color_below_ring': [stalk_color_below_ring],
        'veil_type': [veil_type],
        'veil_color': [veil_color],
        'ring_number': [ring_number],
        'ring_type': [ring_type],
        'spore_print_color': [spore_print_color],
        'population': [population],
        'habitat': [habitat]
    })

    # Make a prediction using the classification model
    prediction = predict_model(classification_model, data=input_data)
    predicted_class = prediction['Label'][0]

    # Redirect to the external site with prediction
    return redirect(f'https://mushrooms-1.onrender.com?prediction={predicted_class}')

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)
