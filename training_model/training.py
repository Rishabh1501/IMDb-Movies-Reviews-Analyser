"""
Copyright (c) 2021 Rishabh Kalra <rishabhkalra1501@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# importing libraries
import os
import pickle
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# importing custom packages
from Logging.logger import Logging


class TrainingAPI:
    """
    Class to Train the model
    
    Keyword arguments:
        cleaned_csv_path=None,
        training_folder_path="Training_Data",
        log_folder_name="Training_Logs",
        log_file_name="3-training_models.txt"
        
    argument --
        cleaned_csv_path: path for the cleaned csv to be used for training
        training_folder_path: Folder path where models are stored.
        log_folder_name: Specifies the folder for Training Logs.
        log_file_name: Specifies the name of the log file.
    
    Return: None
    """
    

    def __init__(self, cleaned_csv_path=None, training_folder_path="Training_Data",
                 log_folder_name="Training_Logs", log_file_name="3-training_models.txt"):

        self.training_folder_path = training_folder_path
        if training_folder_path not in os.listdir():
            os.mkdir(training_folder_path)

        if not cleaned_csv_path:
            self.cleaned_csv_path = os.path.join("Cleaned_Csv_Files","cleaned_data.csv")
        else:
            self.cleaned_csv_path = cleaned_csv_path

        self.log = Logging(os.path.join(log_folder_name, log_file_name))

    def vectorize(self, cleaned_csv_path=None, vector_model_name="vectorize.pickle",
                  vector_save_path=None, folder_save=True):
        """Function to create and save the vector model

        Args:
            cleaned_csv_path (str/path, optional): path to for the cleaned csv to be used for training. Defaults to None.
            vector_model_name (str, optional): Name of the vector model. Defaults to "vectorize.pickle".
            vector_save_path (str/path, optional): path to save vector model. Defaults to None.
            folder_save (bool, optional): Whether to save the model or not, True->save the model, False->don't save the model. Defaults to True.

        Raises:
            Exception: any Exception, check logs for specifics

        Returns:
            sparse matrix: sparse matrix of transformed values
        """
        try:
            self.log.info("Entered function vectorize!")
            if not cleaned_csv_path:
                cleaned_csv_path = self.cleaned_csv_path
                self.log.info("Using Default CSV File")
            else:
                self.log.info("Using User Provided CSV File")

            df = pd.read_csv(cleaned_csv_path)
            df_cols = df.columns
            x = df[df_cols[0]]

            self.log.info("Making a TfidfVectorizer Model")
            vector = TfidfVectorizer()
            self.log.info("Fitting the TfidfVectorizer Model")
            vector.fit(x)

            if folder_save:
                if vector_save_path:
                    save_path = os.path.join(vector_save_path, vector_model_name)
                    self.log.info(f"Saving Model at User defined path: {save_path}")
                else:
                    save_path = os.path.join(self.training_folder_path, vector_model_name)
                    self.log.info(f"Saving Model at Default path: {save_path}")

                with open(save_path, 'wb') as f:
                    pickle.dump(vector, f)
                    self.log.info("Model Saved Successfully!!!")

            x_vector = vector.transform(x)
            self.log.info("Successfully Transformed x!!")
            return x_vector

        except Exception as e:
            self.log.error(f"Function vectorize: {e}")
            raise Exception(e)

    def train_model(self, x_vector, y, train_model_name="svc_model.sav", model_save_path=None, folder_save=True):
        """Function to train the model and save it

        Args:
            x_vector (sparse_matrix): output of the vector model.
            y (pandas.DataFrame): output values for testing.
            train_model_name (str, optional): name of the model. Defaults to "svc_model.sav".
            model_save_path (str/path, optional): path to save the model. Defaults to None.
            folder_save (bool, optional): Whether to save the model or not, True->save the model, False->don't save the model. Defaults to True.

        Raises:
            Exception: any Exception, check logs for specifics

        Returns:
            (str,str): confusion_matrix , classification_report  
        """

        try:
            self.log.info("Entered function train_model")

            x_train, x_test, y_train, y_test = train_test_split(x_vector, y, test_size=0.25, random_state=15)
            self.log.info("Split the data in train and test")

            model = LinearSVC()
            model.fit(x_train, y_train)
            self.log.info("Created the model and fitted it to train data")

            if folder_save:
                if model_save_path:
                    save_path = os.path.join(model_save_path, train_model_name)
                    self.log.info(f"Saving Model at User defined path: {save_path}")
                else:
                    save_path = os.path.join(self.training_folder_path, train_model_name)
                    self.log.info(f"Saving Model at Default Path: {save_path}")

                joblib.dump(model, save_path)
                self.log.info("Model Saved Successfully!!!")

            y_predict = model.predict(x_test)

            cm = confusion_matrix(y_test, y_predict)
            cl_report = classification_report(y_test, y_predict)
            f1_neg = f1_score(y_test, y_predict, pos_label="negative")
            f1_pos = f1_score(y_test, y_predict, pos_label="positive")
            f1 = f1_score(y_test, y_predict, average="weighted")
            self.log.info(
                f"Successfully Predicted with test data: F1 scores-> positive={f1_pos:.2f} negative={f1_neg:.2f} Weighted={f1:.2f}")

            return cm, cl_report

        except Exception as e:
            self.log.error(f"Function train_model: {e}")
            raise Exception(e)
