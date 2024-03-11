from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn .metrics import r2_score,mean_squared_error
import os 
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artefact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting the train and test array')
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],train_arr[:,-1],
                                           test_arr[:,:-1],test_arr[:,-1])
            
            models={
                'RandomForest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbours': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor()


            }
            params={'RandomForest': {'n_estimators':[5,10,15,20,25,30,50,100,200,300],'max_depth':[3,5,10,15,20],'min_samples_leaf':[3,5,10,15,20,25,50,100]},
                    'Decision Tree':{'criterion' : ["squared_error", "friedman_mse", "absolute_error","poisson"],'max_depth':[3,5,10,15,20],'min_samples_leaf':[3,5,10,15,20,25,50,100]} ,
                    'Gradient Boosting':{'learning_rate':[0.1,0.01,0.05,0.001],'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],'n_estimators':[5,10,15,20,25,30,50,100,200,300]},
                    'Linear Regression':{} ,
                    'K-Neighbours':{'n_neighbors':[5,7,9,11]} ,
                    'XGB Regressor':{'learning_rate':[0.1,0.01,0.05,0.001],'n_estimators':[5,10,15,20,25,30,50,100,200,300]},
                    'CatBoost Regressor':{'learning_rate':[0.1,0.01,0.05,0.001],'depth':[6,8,10]} ,
                    'AdaBoost Regressor':{'learning_rate':[0.1,0.01,0.05,0.001],'loss':['linear','square','exponential'],'n_estimators':[5,10,15,20,25,30,50,100,200,300] } }
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            sorted_report=[]
            for w in sorted(model_report,key=model_report.get,reverse=True):
                sorted_report.append((w,model_report[w]))
                
            best_model_name,best_model_score=sorted_report[0]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No model found good enough')
            logging.info('Best found on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            return score



            



        except Exception as e:
            raise CustomException(e,sys)
