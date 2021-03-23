import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from insurancevehicle.Insurancevehicle import InsuranceVehicle

model = pickle.load(  open('C:/Users/Guilherme/Repos/pa004_health_insurance_cross_sell/model/model_lgbm.pkl', 'rb'))
app = Flask (__name__)

@app.route( '/insurancevehicle/ranking', methods =['POST'] )

def insurance_vehicle_ranking():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = Insurancevehicle()       
        data = pipeline.data_cleaning(test_raw)
        data = pipeline.feature_engineering(data)
        data = pipeline.data_preparation(data)
        df_response = pipeline.get_ranking(model, test_raw, data)
        
        return df_response
    
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)