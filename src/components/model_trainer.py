import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainer:
    def initiate_model_trainer(self,train_arr,test_arr):
        self.model_trainer_config=os.path.join('artifacts','model.pkl')
        
        try:
            logging.info("Splitting Data into training data and testing data")
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],
                                           train_arr[:,-1],
                                           test_arr[:,:-1],
                                           test_arr[:,-1])
            models={"LogisticRegression":LogisticRegression(),
                    "DecisionTreeClassifier":DecisionTreeClassifier(),
                    "RandomForestClassifier":RandomForestClassifier(),
                    }
            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info("Model Training has been done and now selecting the best model")
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise Custom_Exception("No best Model Found")
            logging.info("Best Model has been found and the model is '{}'".format(best_model_name))

            predicted=best_model.predict(X_test)
            score=accuracy_score(y_test,predicted)
            logging.info("Since the best model has been  discovered now combining the training data and test data into one complete dataset before pushing for the Model Deployment")
            arr1=np.vstack((X_train,X_test))
            arr2=np.vstack((y_train.reshape(-1,1),y_test.reshape(-1,1)))
            logging.info("Concatention of array datapoints from training data and test data has been completed")
            best_model.fit(arr1,arr2)
            logging.info("Best model has been fit on the complete data")
            save_object(file_path=self.model_trainer_config,obj=best_model)
            logging.info("Best model has been saved in artifact folder with path as '{}'".format(self.model_trainer_config))
            print(best_model_name)
            return score
        except Exception as e:
            raise Custom_Exception(e,sys)