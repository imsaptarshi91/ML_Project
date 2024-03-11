import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artefact/model.pkl'
            preprocessor_path='artefact/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)





class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,test_preparation_course,
                 lunch,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.test_preparation_course=test_preparation_course
        self.parental_level_of_education=parental_level_of_education

        self.lunch=lunch
        self.reading_score=reading_score
        self.writing_score=writing_score
    def get_data_as_frame(self):
        try:
            custom_data_input_dict={
                'gender':[self.gender],
                'race/ethnicity':[self.race_ethnicity],
                'parental level of education':[self.parental_level_of_education],
                'test preparation course':[self.test_preparation_course],
                'lunch':[self.lunch],
                'reading score':[self.reading_score],
                'writing score':[self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
