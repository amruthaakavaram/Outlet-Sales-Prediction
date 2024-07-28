from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('catboost_regressor_model.pkl')

# Function to handle one-hot encoding for categorical variables
def preprocess_input(data):
    item_fat_content = ['Low Fat', 'Non Edible', 'Regular']
    item_type = ['New_Drinks', 'New_Foods', 'New_Non Consumables']
    outlet_size = ['High', 'Medium', 'Small', 'Unknown']
    outlet_location_type = ['Tier 1', 'Tier 2', 'Tier 3']
    outlet_type = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']

    # One-hot encoding
    data_encoded = []

    data_encoded.append(float(data['Item_Weight']))
    data_encoded.append(float(data['Item_Visibility']))
    data_encoded.append(float(data['Item_MRP']))
    data_encoded.append(float(data['Outlet_Establishment_Year']))
    data_encoded.append(float(data['Years_of_Operation']))

    data_encoded.extend([1 if data['Item_Fat_Content'] == fat else 0 for fat in item_fat_content])
    data_encoded.extend([1 if data['Item_Type'] == type_ else 0 for type_ in item_type])
    data_encoded.extend([1 if data['Outlet_Size'] == size else 0 for size in outlet_size])
    data_encoded.extend([1 if data['Outlet_Location_Type'] == location else 0 for location in outlet_location_type])
    data_encoded.extend([1 if data['Outlet_Type'] == type_ else 0 for type_ in outlet_type])

    return np.array(data_encoded).astype(np.float32).reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input from the form
        input_data = {
            'Item_Weight': request.form.get('Item_Weight'),
            'Item_Visibility': request.form.get('Item_Visibility'),
            'Item_MRP': request.form.get('Item_MRP'),
            'Outlet_Establishment_Year': request.form.get('Outlet_Establishment_Year'),
            'Years_of_Operation': request.form.get('Years_of_Operation'),
            'Item_Fat_Content': request.form.get('Item_Fat_Content'),
            'Item_Type': request.form.get('Item_Type'),
            'Outlet_Size': request.form.get('Outlet_Size'),
            'Outlet_Location_Type': request.form.get('Outlet_Location_Type'),
            'Outlet_Type': request.form.get('Outlet_Type'),
        }

        # Preprocess the input data
        input_data_np = preprocess_input(input_data)

        # Make a prediction
        prediction = model.predict(input_data_np)

        # Redirect to result page with prediction result as URL parameter
        return redirect(url_for('result', prediction=prediction[0]))

    # Render index.html on GET request
    return render_template('index.html')

@app.route('/result')
def result():
    # Get prediction result from URL parameter
    prediction = request.args.get('prediction')

    # Render result.html with prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
