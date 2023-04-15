import pickle

with open('rf_model.pkl', 'rb') as file:
    rf = pickle.load(file)

with open('label_encoder_state.pkl', 'rb') as file:
    label_encoder_state = pickle.load(file)

with open('label_encoder_owner_types.pkl', 'rb') as file:
    label_encoder_owner_types = pickle.load(file)

with open('label_encoder_project_types.pkl', 'rb') as file:
    label_encoder_project_types = pickle.load(file)

from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data from the frontend

    grossFloorArea = float(request.form['grossFloorArea'])
    state = request.form['state']
    ownerTypes = request.form['ownerTypes']
    projectTypes = request.form['projectTypes']

    # Create a new dataframe with the form data
    input_data = pd.DataFrame({'GrossFloorArea': [grossFloorArea],
                               'State': [state],
                               'OwnerTypes': [ownerTypes],
                               'ProjectTypes': [projectTypes]
                               })
    

 # Apply label encoding using label_encoder for State, OwnerTypes, and ProjectTypes columns
    input_data['State'] =  label_encoder_state.transform(input_data['State'])
    input_data['OwnerTypes'] = label_encoder_owner_types.transform(input_data['OwnerTypes'])
    input_data['ProjectTypes'] = label_encoder_project_types.transform(input_data['ProjectTypes'])

   # Convert input_data to dictionary
    # input_data_dict = input_data.to_dict()


    # print(input_data)

    # # Make predictions using the trained model
    certLevel = rf.predict(input_data)[0]

    # Render the prediction result in a new page
    return render_template('result.html', prediction= certLevel)

if __name__ == '__main__':
    app.run(debug=True)