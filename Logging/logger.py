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
from datetime import datetime

class Logging:
    """
    Class to log the progress of the project
    
    Keyword arguments: None
    
    argument -- 
        file_path : location of the log file
    
    Return: None 
    """
    
    
    def __init__(self, file_path):
        self.file_path = file_path

    def __repr__(self):
        return f"Logging({self.file_path})"

    def log(self, message, log_type):
        """
        custom log method
        
        Args:
            message -> str: message to be logged
            log_type -> str: type of message to be logged ,for example = ("info","error", etc..)
        """   
             
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")

    def info(self, message, log_type="INFO"):
        """
        Info log method
        
        Args:
            message -> str: message to be logged
            log_type -> str: INFO 
        """   
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")

    def warning(self, message, log_type="WARNING"):
        """
        Warning log method
        
        Args:
            message -> str: message to be logged
            log_type -> str: WARNING 
        """   
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")

    def error(self, message, log_type="ERROR"):
        """
        Error log method
        
        Args:
            message -> str: message to be logged
            log_type -> str: ERROR 
        """   
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")


if __name__ == "__main__":
    logger = Logging("test1.txt")
    logger.log("This is for testing - 1", "Custom text")
    logger.info("This is for testing")
    logger.warning("This is for testing")
    logger.error("This is for testing")
