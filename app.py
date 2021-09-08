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
import pandas as pd
from flask import Flask, request, render_template

# importing custom packages
from data_import.data_input import DataInput
from data_cleaning.data_cleaning import Cleaner
from training_model.training import TrainingAPI
from predicting_model.prediction import PredictAPI
from email_yagmail.email_bot_using_yagmail import email_send

#making an instance of app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


#API to train the model
@app.route('/train', methods=["GET", "POST"])
def train():
    data_input = DataInput()
    data_cleaning = Cleaner()
    csv_path = os.path.join("Cleaned_Csv_Files", "cleaned_data.csv")
    if "Cleaned_Csv_Files" not in os.listdir():
        os.mkdir("Cleaned_Csv_Files")

    if request.method == "POST":
        json_data = request.json()
        df = data_input.ret_dataframe(json_data["input_csv_path"])
        df_cleaned = data_cleaning.ret_cleaned_dataframe(df)
        data_cleaning.save_dataframe_in_csv(df_cleaned, csv_path)

        del df
        del df_cleaned

        df_cleaned = pd.read_csv(csv_path)
        y = df_cleaned["sentiment"]

        train = TrainingAPI(cleaned_csv_path=csv_path)
        x_vector = train.vectorize()
        cm, cl_report = train.train_model(x_vector, y)

        print(cm)
        print(cl_report)


#API to predict
@app.route('/predict', methods=["POST", "GET"])
def predict():
    prediction = PredictAPI()

    if request.method == "POST":    
        if request.json:
            data = request.json
            review = data["review"]
        elif request.form:
            data = dict(request.form)
            review = data["sentence_form"]
            # return review
        else:
            return "nothing_happened"

        print("Review:",review)
        clean_sentence = prediction.clean_sentence(review)
        predicted_data = prediction.predict_model_sentence(clean_sentence)
        return render_template("index.html",review=review,prediction=predicted_data)
    
        
@app.route('/contact',methods=['POST'])
def contact():
    if request.method == "POST":    
        if request.form:
            data = request.form
            print(data)
            to_email = data["toemail"]
            message = data["message"]
            email_send(to_email,message)
        else:
            print("Nothing happened")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)