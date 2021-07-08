from datetime import datetime


class Logging:

    def __init__(self, file_path):
        self.file_path = file_path

    def log(self, message, log_type):
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")

    def info(self, message, log_type="INFO"):
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")

    def warning(self, message, log_type="WARNING"):
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")

    def error(self, message, log_type="ERROR"):
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        with open(self.file_path, "a+") as f:
            f.write(date_time + " -> " + log_type + ": " + message + "\n")


if __name__ == "__main__":
    logger = Logging("test1.txt")
    logger.log("This is for testing - 1", "Custom text")
    logger.info("This is for testing")
    logger.warning("This is for testing")
    logger.error("This is for testing")
