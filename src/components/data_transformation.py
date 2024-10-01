## Featuring engg and feature selection

import os,sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            numerical_feature=['writing_score','reading_score']
            categorical_feature=['gender', 'race_ethnicity',
                                  'parental_level_of_education',
                                    'lunch', 'test_preparation_course',
                                    ]
            logging.info("---> Pipeline Started <---")
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                     ('scalar',StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder",OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("---> Numerical Columns Scaling Complted <---")
            logging.info("---> Categorical Columns Encoding Complted <---")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipelin",num_pipeline,numerical_feature),
                    ("cat_pipeline",cat_pipeline,categorical_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("---> Reading Train and Test data <---")

            logging.info("---> Obtaining preprocessor objedt <---")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_feature=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f" ---> Applying preprocessing object on training dataframe and testing dataframe <---")
            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessed Object")

            save_object(

                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            ) 

        except Exception as e:
            raise CustomException(e,sys)
        
















