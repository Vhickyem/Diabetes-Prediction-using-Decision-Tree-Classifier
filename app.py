import numpy as np
from flask import Flask, request, render_template
import pickle

# create an app object using the flask class
app = Flask(__name__)

# load the trained model from the pickle file
with open('models/diabetes_decision_tree.pkl', 'rb') as file:
    model = pickle.load(file)

# define the route for the home page

@app.route('/')
def home():
    return render_template('index.html')

# define the route for prediction

@app.route('/predict', methods=['POST'])
def predict():
    # get the input values from the form
    age = int(request.form.get('Age'))
    gender = 1 if request.form.get('Gender') == 'Male' else 0
    polyuria = 1 if request.form.get('Polyuria') == 'Yes' else 0
    polydipsia = 1 if request.form.get('Polydipsia') == 'Yes' else 0
    sudden_weight_loss = 1 if request.form.get('sudden weight loss') == 'Yes' else 0
    weakness = 1 if request.form.get('weakness') == 'Yes' else 0
    polyphagia = 1 if request.form.get('Polyphagia') == 'Yes' else 0
    genital_thrush = 1 if request.form.get('Genital thrush') == 'Yes' else 0
    visual_blurring = 1 if request.form.get('visual blurring') == 'Yes' else 0
    itching = 1 if request.form.get('Itching') == 'Yes' else 0
    irritability = 1 if request.form.get('Irritability') == 'Yes' else 0
    delayed_healing = 1 if request.form.get('delayed healing') == 'Yes' else 0
    partial_paresis = 1 if request.form.get('partial paresis') == 'Yes' else 0
    muscle_stiffness = 1 if request.form.get('muscle stiffness') == 'Yes' else 0
    alopecia = 1 if request.form.get('Alopecia') == 'Yes' else 0
    obesity = 1 if request.form.get('Obesity') == 'Yes' else 0
    
    # create a numpy array with the input values
    input_data = np.array([[age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, genital_thrush, visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity]])
    
    # make the prediction
    prediction = model.predict(input_data)[0]
    
    # Return the result
    result = "Diabetes detected" if prediction == 1 else "No diabetes detected"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)