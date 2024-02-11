from flask import Flask,request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Pipeline.predict_pipeline import CustomData, PredictPipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('prediction.html')

@app.route('/predictdata',methods=['GET',"POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('prediction.html')
    else:
        data=CustomData(
            ram=request.form.get('ram'),
            screen_size=request.form.get('screen_size'),
            battery_power=request.form.get('battery_power'),
            int_memory=request.form.get('int_memory'),
            talk_time=request.form.get('talk_time'),
            n_cores=request.form.get('n_cores'),
            blue=request.form.get('blue'),
            pc=request.form.get('pc')            
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        message=""
        if results==0:
            message="the phone may sell at lower cost"
        elif results==1:
            message="the phone may sell at medium cost"
        else:
            message="the phone may sell at higher cost"
        return render_template('prediction.html',results=message)
    
if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080)