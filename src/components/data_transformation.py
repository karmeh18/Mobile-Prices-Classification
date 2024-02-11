import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import Custom_Exception
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformation:
    def get_data_transformation_object(self):
        self.get_data_transformation_config=os.path.join('artifacts','preprocessor.pkl')
        """This function is responsible for Data Transformation
        """
        try:
            num_columns=['ram', 'screen_size', 'battery_power', 'int_memory', 'talk_time', 'n_cores', 'blue', 'pc']
            num_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='mean'))
                ]
            )
            logging.info("Pipeline has been created for the columns {}".format(num_columns))
            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,num_columns)]
            )
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e,sys)
        

    def initiate_data_transformation(self,raw_data_path_directory):
        try:
            raw_data=pd.read_csv(raw_data_path_directory)
            logging.info("Training Data and Testing data has been created")
            logging.info("Obtaining Preprocess Object")

            target_column='price_range'
            X=raw_data.loc[:,['ram', 'px_height', 'px_width', 'sc_h', 'sc_w', 'battery_power', 'int_memory', 'talk_time', 'n_cores', 'blue', 'pc']]
            y=raw_data[target_column]

            logging.info('Feature Engineering Initiated')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Feature Engineering - Formation of Screen width and Screen Height in both training and testing data has started")
            X_train['screen_height']=X_train['px_height']/96
            X_train['screen_width']=X_train['px_width']/96
            X_test['screen_height']=X_test['px_height']/96
            X_test['screen_width']=X_test['px_width']/96

            logging.info("Feature Engineering - Formation of Screen size has been completed in both traning and testing data has started")
            X_train['screen_size']=np.sqrt(X_train['screen_height']**2 + X_train['screen_width']**2)
            X_test['screen_size']=np.sqrt(X_test['screen_height']**2 + X_test['screen_width']**2)
            original_columns=['px_height','px_width','screen_height','screen_width']

            logging.info('Feature Engineering - Original columns {} have been deleted '.format(original_columns))
            X_train=X_train.drop(original_columns,axis=1)
            X_test=X_test.drop(original_columns,axis=1)

            logging.info("Applying preprocessing object to training and test data")
            preprocessor_obj=self.get_data_transformation_object()
            input_feature_train_arr=preprocessor_obj.fit_transform(X_train)
            input_feature_test_arr=preprocessor_obj.fit_transform(X_test)

            logging.info("Applied preprocessor object to training data and test data ")

            logging.info("Combining Preprocessed Input datapoints from training and test data with target feature of training and test data")
            train_arr=np.c_[input_feature_train_arr,np.array(y_train)]
            test_arr=np.c_[input_feature_test_arr,np.array(y_test)]

            logging.info("Preprocessed and Concatenation has been completed")

            save_object(file_path=self.get_data_transformation_config,obj=preprocessor_obj)
            return (train_arr,  
                    test_arr)
        except Exception as e:
            raise Custom_Exception(e,sys)
        






    
