import sys
import os
import datetime

class Logger():
    def __init__(self, file_path):
        name = "train_log_" + str(datetime.datetime.now()).replace(":", "").replace(" ", "_") + ".txt"
        filename = os.path.join(file_path, name)
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass