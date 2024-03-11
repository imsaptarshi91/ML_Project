import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os 
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artefact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns=['reading score', 'writing score']
            cat_columns=['gender','race/ethnicity','parental level of education',
                           'lunch','test preparation course']
            num_pipeline= Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median')),
                       ('scaler',StandardScaler())]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Numerical columns:{numerical_columns}')
           
            logging.info(f'Categorical columns:{cat_columns}')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,cat_columns)
                ]

            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reading the train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_object()
            target_column='math score'
            numerical_columns=['reading score', 'writing score']
            target_train_df=train_df[target_column]
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)


            target_test_df=test_df[target_column]
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)

            logging.info('Applying preprocessing object on training and test dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
           
            logging.info('Saved preprocessing object')
            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
        
        

            
        except Exception as e:
            raise CustomException(e,sys)