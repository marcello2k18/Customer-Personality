from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Load the model
classifier = pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)

# Customer segmentation function
def segment_customers(input_data):
    prediction = classifier.predict(
        pd.DataFrame([input_data], columns=['Income', 'Kidhome', 'Teenhome', 'Age', 'Partner', 'Education_Level'])
    )
    return prediction[0]

# Define the route to render the form and handle prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form inputs
        income = float(request.form['income'])
        num_kids = int(request.form['num_kids'])
        num_teens = int(request.form['num_teens'])
        age = int(request.form['age'])
        partner = request.form['partner']
        education_level = request.form['education']

        # Prepare input data for prediction
        input_data = {
            'Income': income,
            'Kidhome': num_kids,
            'Teenhome': num_teens,
            'Age': age,
            'Partner': partner,
            'Education_Level': education_level
        }

        # Perform segmentation and get prediction
        prediction = segment_customers(input_data)

        # Map numeric prediction to readable label
        cluster_map = {
            0: 'Cluster 0',
            1: 'Cluster 1',
            2: 'Cluster 2',
            3: 'Cluster 3'
        }

        predicted_cluster = cluster_map[prediction]

        # Return prediction result as JSON to the frontend
        return jsonify({'prediction_result': predicted_cluster})

    return render_template('index.html')  # Ensure the form is in the 'index.html' page

if __name__ == '__main__':
    app.run(debug=True)
