from flask import Flask, request, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/mushroom_classification_model')

# Define the home route
@app.route('/')
def home():
    return render_template('jess.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    input_data = {
        'cap-shape': request.form.get('cap_shape', ''),
        'cap-surface': request.form.get('cap_surface', ''),
        'cap-color': request.form.get('cap_color', ''),
        'bruises': request.form.get('bruises', ''),  # Use get to avoid KeyError
        'odor': request.form.get('odor', ''),
        'gill-attachment': request.form.get('gill_attachment', ''),
        'gill-spacing': request.form.get('gill_spacing', ''),
        'gill-size': request.form.get('gill_size', ''),
        'gill-color': request.form.get('gill_color', ''),
        'stalk-shape': request.form.get('stalk_shape', ''),
        'stalk-root': request.form.get('stalk_root', ''),
        'stalk-surface-above-ring': request.form.get('stalk_surface_above_ring', ''),
        'stalk-surface-below-ring': request.form.get('stalk_surface_below_ring', ''),
        'stalk-color-above-ring': request.form.get('stalk_color_above_ring', ''),
        'stalk-color-below-ring': request.form.get('stalk_color_below_ring', ''),
        'veil-type': request.form.get('veil_type', ''),
        'veil-color': request.form.get('veil_color', ''),
        'ring-number': request.form.get('ring_number', ''),
        'ring-type': request.form.get('ring_type', ''),
        'spore-print-color': request.form.get('spore_print_color', ''),
        'population': request.form.get('population', ''),
        'habitat': request.form.get('habitat', '')
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Generate predictions
    predictions = predict_model(model, data=input_df)
    prediction = predictions['prediction_label'][0]

    # Return the result
    return render_template('jess.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
