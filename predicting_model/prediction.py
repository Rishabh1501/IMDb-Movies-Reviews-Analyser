# importing libraries
import os
import pickle
import joblib
import numpy as np
import pandas as pd

# importing custom packages
from Logging.logger import Logging
from data_import.data_input import DataInput
from data_cleaning.data_cleaning import Cleaner


class PredictAPI:

    def __init__(self, prediction_folder_path="Prediction_Data", training_folder_path="Training_Data",
                 log_folder_name="Prediction_Logs", log_file_name="3-prediction.txt"):
        self.prediction_folder_path = prediction_folder_path
        self.training_folder_path = training_folder_path
        self.data_input = DataInput(log_folder_name, "1-file_input.txt")
        self.cleaner = Cleaner(log_folder_name, "2-data_cleaning.txt")

        if prediction_folder_path not in os.listdir():
            os.mkdir(prediction_folder_path)

        self.log = Logging(os.path.join(log_folder_name, log_file_name))

    def clean_sentence(self, sentence):
        try:
            sentence = self.cleaner.review_to_words(sentence)
            self.log.info("Sentence Cleaned!!")
            return sentence
        except Exception as e:
            self.log.error(f"function clean_sentence: {e}")
            raise Exception(e)

    def clean_csv_data(self, csv_path):
        try:
            df = self.data_input.ret_dataframe(csv_path)
            cleaned_df = self.cleaner.ret_cleaned_dataframe(df)
            self.log.info("CSV Cleaned!!")
            return cleaned_df
        except Exception as e:
            self.log.error(f"function clean_csv_data: {e}")
            raise Exception(e)

    def predict_model_csv(self, dataframe, model_name="model.sav", vector_model="vectorize.pickle"):
        try:
            vector_path = os.path.join(self.training_folder_path, vector_model)
            model_path = os.path.join(self.training_folder_path, model_name)
            with open(vector_path, 'rb') as f:
                vector = pickle.load(f)
                self.log.info("Vector model loaded Successfully!!")

            nb_model = joblib.load(model_path)
            self.log.info("Naive Bayes model loaded Successfully!!")
            df_cols = dataframe.columns
            x = dataframe[df_cols[0]]
            x_vector = vector.transform(x)
            self.log.info("Data Vector Transformation Successful!!")

            y_predict = nb_model.predict(x_vector)
            self.log.info("Prediction Successful!!")

            dataframe["sentiment"] = y_predict
            csv_save_path = os.path.join(self.prediction_folder_path, "Prediction.csv")
            dataframe.to_csv(csv_save_path, index_label=False)
            self.log.info(f"Successfully Saved the Prediction file at location: {csv_save_path}")

        except Exception as e:
            self.log.error(f"function predict_model_csv: {e}")
            raise Exception(e)

    def predict_model_sentence(self, sentence, model_name="svc_model.sav", vector_model="vectorize.pickle"):
        try:
            df = pd.DataFrame([sentence], columns=['review'])

            vector_path = os.path.join(self.training_folder_path, vector_model)
            model_path = os.path.join(self.training_folder_path, model_name)
            with open(vector_path, 'rb') as f:
                vector = pickle.load(f)
                self.log.info("Vector model loaded Successfully!!")

            nb_model = joblib.load(model_path)
            self.log.info("Naive Bayes model loaded Successfully!!")

            x_vector = vector.transform(df['review'])
            # print(x_vector)
            y_predict = nb_model.predict(x_vector)
            print(y_predict)
            # predicted_probability = nb_model.predict_proba(x_vector)
            # if list(predicted_probability.flatten())[0] == list(predicted_probability.flatten())[1]:
            #     print("UNKNOWN")
            # elif list(predicted_probability.flatten())[np.argmax(predicted_probability)] > 0.6:
            # else:
            #     print("UNKNOWN")

            self.log.info("Prediction Successful!!")

        except Exception as e:
            self.log.error(f"function predict_model_sentence: {e}")
            raise Exception(e)
