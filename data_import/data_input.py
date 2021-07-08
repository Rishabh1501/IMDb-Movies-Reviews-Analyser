import os
import pandas as pd
from Logging.logger import Logging


class DataInput:

    def __init__(self, log_folder_name="Training_Logs", log_file_name="1-file_input.txt"):
        self.log = Logging(os.path.join(log_folder_name, log_file_name))

    def ret_dataframe(self, file_path, rows=None, drop_null=True):
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

            return df

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
