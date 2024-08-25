from flask import Flask, render_template, request
from pycaret.regression import load_model, predict_model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('models/regression_model')


# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('ryan.html')


# Define a route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
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
    predictions = predict_model(model, data=input_data)

    # Extract the prediction value
    result = predictions['prediction_label'][0]

    # Return the prediction and render it in the template
    return render_template('ryan.html', prediction=result)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
