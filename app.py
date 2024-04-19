from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
from waitress import serve

app = Flask(__name__)

@app.route('/app')
def frontpage():
    return render_template('index.html')

# Route to handle POST requests to '/api'
@app.route('/app/api', methods=['POST'])
def process_api_request_student():
    data = request.get_json()
    merchant=data['merchant']
    city=data['city']
    amount=float(data['amount'])
    status=model(merchant,city,amount)
    return jsonify({'message': status})

mercs=['khaadi', 'sapphire', 'outfitters', 'sana safinaz', 'gul ahmed', 'bachaa party']
mercs.sort()
cities=['lahore', 'multan', 'karachi', 'rawalpindi', 'faisalabad', 'islamabad','sukkur', 'gujranwala', 'taxila', 'hyderabad', 'sahiwal', 'qasim pur','bahawalpur', 'peshawar', 'wah cantt', 'quetta', 'sialkot', 'jehangira','sargodha' ,
        'talagang', 'wazirabad', 'kamoke', 'khurrianwala', 'shabqadar','tando allahyar', 'wagah', 'rahim yar khan', 'nawabshah', 'muzaffarabad', 'gujrat', 'sadiqabad', 'kotri', 'jamshoro', 'tando jam', 'kunjah', 'jhang','matiari ', 'abbottabad']   
cities.sort()

def encode_user_input(city, merchant):
    global mercs
    global cities

    df=pd.read_csv('processed.csv')

    new_row = pd.Series({'Merchant': merchant, 'Destination City': city})

    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df=pd.get_dummies(df,columns=['Merchant','Destination City'])

    return df.iloc[-2:]

def model(merchant, city, amount):
    global mercs
    global cities

    merchant=merchant.lower().strip()
    city=city.lower().strip()
    
    if amount<0:
        return "Enter valid Amount"
    elif merchant not in mercs:
        return "Merchant not found. Valid Merchants: Khaadi, Sapphire, Outfitters, Sana Safinaz, Gul Ahmed, Bachaa Party"
    elif city not in cities:
        return "City not found"
    amount=amount**(1/5)


    X=encode_user_input(city, merchant)
    X['Amount']=amount
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        X=scaler.transform(np.array(X))

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    pred=model.predict(X)
    prediction=int(pred[-1])
    print(prediction)
    if prediction==0:
        return "Delivered"
    elif prediction==1:
        return "Returned"
    else:
        return "Lost"   

if __name__ == '__main__':
    serve(app)


#if __name__ == '__main__':
#    app.run(debug=True)