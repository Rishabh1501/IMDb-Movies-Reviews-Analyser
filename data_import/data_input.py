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

#importing libraries
import os
import pandas as pd
from Logging.logger import Logging


class DataInput:
    """    
    Class to take input of CSV files for training and prediction
    
    Keyword arguments: log_folder_name="Training_Logs", log_file_name="1-file_input.txt"
    
    argument -- 
        log_folder_name: Specifies the folder for Training Logs
        log_file_name: Specifies the name of the log file
    
    Return: None
    """   

    def __init__(self, log_folder_name="Training_Logs", log_file_name="1-file_input.txt"):
        self.log = Logging(os.path.join(log_folder_name, log_file_name))

    def ret_dataframe(self, file_path:"str", rows:"int"=None, drop_null:"bool"=True) -> "pd.DataFrame":
        """
        returns a DataFrame after reading data from a given CSV file
        
        Args:
            file_path -> str/path : path of the CSV file.
            rows (int, optional): numbers of rows to read. Defaults to None: reads all rows.
            drop_null (bool, optional):Whether to include or exclude null_values in DataFrame. Defaults to True.

        Raises:
            OSError: path not correct
            Exception: any other Exception

        Returns:
            pandas.DataFrame: pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path, nrows=rows)
            self.log.info("DataFrame created successfully!")
            null_check = df.isna().values.any()  # False for no null and True if null Present

            # if else ladder for precise Logging and null values drop
            if drop_null and null_check:
                df.dropna(inplace=True)
                self.log.info("Null Values Present and Dropped!")
            elif drop_null and not null_check:
                self.log.info("No Null Values Present!")
            elif not drop_null and null_check:
                self.log.warning("No action taken regarding Null Values!")
            elif not null_check:
                self.log.info("No Null Values Present!")

            return df #returning the DataFrame

        except OSError as e:
            self.log.error(f"File Not Found!! function ret_dataframe: {e}")
            raise OSError(e)

        except Exception as e:
            self.log.error(e)
            raise Exception(f"function ret_dataframe: {e}")


if __name__ == "__main__":
    file = DataInput("..\IMDB Dataset.csv", "..\Training_Logs")
    df = file.ret_dataframe(10, drop_null=False)
    print(df)
