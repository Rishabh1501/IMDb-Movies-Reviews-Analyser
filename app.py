# importing libraries
import os
import pandas as pd
from flask import Flask, request

# importing custom packages
from data_import.data_input import DataInput
from data_cleaning.data_cleaning import Cleaner
from training_model.training import TrainingAPI
from predicting_model.prediction import PredictAPI

app = Flask(__name__)


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


@app.route('/predict', methods=["POST", "GET"])
def predict():
    prediction = PredictAPI()

    if request.method == "POST":
        json_data = request.json
        review = json_data["review"]
        clean_sentence = prediction.clean_sentence(review)
        predicted_data = prediction.predict_model_sentence(review)
        return predicted_data
    # positive = """I was honestly expecting much worse than what I received after watching this movie. I figured it was going to be supper cheesy and dumb. But it wasn't. It wasn't amazing. It wasn't perfect. But it was entertaining. And that's really all that matters when watching a movie isn't it? I enjoyed this movie, and you likely will as well if you are into sci-fi alien action movies."""
    #
    # negative = """That feeling you get after finding movie with okay reviews and giving a chance...only to discover that it's soulless garbage?
    # And then the generic paid for positive reviews confirm your suspicion of the Hollywood hype machine masquerading as everyday Joe..
    # "Just shut your brain off and enjoy your mind trash..."
    # "Not sure what all the negative reviews are about. It was super fun and enjoyable...give it a chance"
    # On the bright side, at least 10% of humanity hasn't yet been assimilated by the Borg."""
    #
    # negative2 = """Better watch Edge of Tomorrow again.
    # With this film, you don't just have to turn your brain off, you would need a Lobotomy."""
    #
    # # sentence = "Bad Movie"
    #
    # review = """The movie fees cheap and not necessarily to do with the special affects budget. It also feels like it was made by a committee, and is what leads to it ultimately falling flat, with too many convenient potlines that just arise to serve the next spectacle. You have the convenient smart people who solve problems when you would least expect it, convenient comedic relief that is totally unnecessary but is there because polls tell you there needs to be some light humour in this.
    # It doesn't feel original, there is no perceived threat for the main character, and for all of the similarities that it has with perhaps say The Edge Of Tomorrow, it has none of the cohesion, smarts, real-world physics or tension that everything will be okay. This is a movie that will be very easy to forget."""

if __name__ == "__main__":
    app.run(debug=True)