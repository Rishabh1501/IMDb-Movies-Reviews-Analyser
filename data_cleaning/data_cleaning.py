import os
import nltk
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from Logging.logger import Logging


# nltk.download()

class Cleaner:

    def __init__(self,log_folder_name="Training_Logs", log_file_name="2-data_cleaner.txt"):
        self.stemmer = PorterStemmer()
        self.stop_words = stopwords.words('english')
        self.unnecessary_words = ["br", "'ll", "..", "....", "n't", "...", " ... "]
        self.punctuation = string.punctuation
        self.log = Logging(os.path.join(log_folder_name, log_file_name))

    def review_to_words(self, sentence):
        try:
            words = nltk.word_tokenize(sentence)
            words_list = list()
            for word in words:
                word = word.lower()
                letter_list = list()
                # print(word)
                if word not in self.stop_words:
                    if word not in self.unnecessary_words:
                        for letter in word:
                            if letter not in self.punctuation:
                                letter_list.append(letter)
                        if letter_list:
                            word = ''.join(letter_list)
                            words_list.append(self.stemmer.stem(word))
            return " ".join(words_list)

        except Exception as e:
            self.log.error(f"function review_to_words: {e}")
            raise Exception(e)

    def ret_cleaned_dataframe(self, dataframe, col_num=0):
        try:
            col = dataframe.columns
            self.log.info(f"Columns extracted, cleaning column '{col[col_num]}' for processing!")
            dataframe[col[col_num]] = dataframe[col[col_num]].apply(self.review_to_words)
            # dataframe[col[col_num+1]] = dataframe[col[col_num+1]].apply(lambda x: 1 if x == "positive" else 0)
            self.log.info(f"Column '{col[col_num]}' Cleaned Successfully!")
            return dataframe

        except Exception as e:
            self.log.error(f"function ret_cleaned_dataframe: {e}")
            raise Exception(e)

    def save_dataframe_in_csv(self,dataframe,file_path):
        try:
            dataframe.to_csv(file_path,index_label=False)
            self.log.info("DataFrame converted to CSV Successfully!!")
        except Exception as e:
            self.log.error(f"function save_dataframe_in_csv: {e}")
            raise Exception(e)


