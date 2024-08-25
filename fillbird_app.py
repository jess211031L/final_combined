from flask import Flask, request, render_template
from pycaret.anomaly import load_model, predict_model
from joblib import load
import pandas as pd
import zipfile
import os


# zip_file_path = 'anomaly_detection_model.zip'
# extracted_dir_path = 'anomaly_detection_model'

# # Unzip the model file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extracted_dir_path)

# # Load the unzipped model
# model_path = os.path.join(extracted_dir_path, 'anomaly_detection_model')


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/anomaly_detection_model')

# Define the home route
@app.route('/')
def home():
    return render_template('fillbird.html')

# Define the prediction route
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

    # Print the columns for debugging
    print(f"Input DataFrame columns: {input_df.columns}")

    # Check for missing or invalid data
    # if input_df.isna().any().any():
    #     prediction = "Invalid input: Please ensure all fields are filled correctly."
    # else:
       # Generate predictions
    predictions = predict_model(model, data=input_df)
    anomaly_flag = predictions['Anomaly_Score'][0]

    # Convert numerical prediction to "Anomaly" or "Not Anomaly"
    prediction = "Anomaly" if anomaly_flag > 0 else "Not Anomaly"
    
        # prediction = f"An error occurred during prediction: {str(e)}"

    # Return the result to the template
    return render_template('fillbird.html', prediction=prediction)


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)