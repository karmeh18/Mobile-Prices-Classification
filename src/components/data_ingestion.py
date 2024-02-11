import os
import sys
import pandas as pd

from src.exception import Custom_Exception
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestionConfig:
    train_data_path=os.path.join("artifacts",'train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into the Data Ingestion stage")
        try:
            df=pd.read_csv("notebook\data\mobile_price_classification.csv")
            logging.info("Data Importing has been completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Training Data directory has been created and an csv file has been exported to the path '{}'".format(self.ingestion_config.train_data_path))

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Testing Data directory has been created and an csv file has been exported to the path '{}'".format(self.ingestion_config.test_data_path))

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw Data directory has been created and an csv file has been exported to the path '{}'".format(self.ingestion_config.raw_data_path))

            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise Custom_Exception(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()

    transformation=DataTransformation()
    train_arr,test_arr=transformation.initiate_data_transformation(raw_data_path)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))



