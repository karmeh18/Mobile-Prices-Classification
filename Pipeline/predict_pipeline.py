import sys
import os
import pandas as pd

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts",'model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info('Model and preprocessor object has been loaded')

            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)
        
class CustomData:
    def __init__(self,
                 ram,
                 screen_size,
                 battery_power,
                 int_memory,
                 talk_time,
                 n_cores,
                 pc,
                 blue
                 ):
         self.ram=ram
         self.screen_size=screen_size
         self.battery_power=battery_power
         self.int_memory=int_memory
         self.talk_time=talk_time
         self.n_cores=n_cores
         self.blue=blue
         self.pc=pc
         
    def get_data_as_dataframe(self):
        try:
            #['ram', 'screen_size', 'battery_power', 'int_memory', 'talk_time', 'n_cores', 'blue', 'pc']
            custom_data_input_dict={
                'ram':[self.ram],
                'screen_size':[self.screen_size],
                'battery_power':[self.battery_power],
                'int_memory':[self.int_memory],
                "talk_time":[self.talk_time],
                "n_cores":[self.n_cores],
                "blue":[self.blue],
                "pc":[self.pc]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Custom_Exception(e,sys)
        