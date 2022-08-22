import os
import sys
import time


class Logger(object):
    def __init__(self, log_name, include_timestamp=False, log_dir="logs",
                 log_file_mode="w+"):
        self.console = sys.stdout
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        timestamp = time.strftime("%Y%M%d-%H%M%S", time.localtime())
        self.log_file = open(os.path.join(
            log_dir,
            log_name + "-" + timestamp + ".log"
            if include_timestamp else log_name + ".log"),
            log_file_mode)
        if include_timestamp:
            self.console.write(timestamp + "\n")
            self.log_file.write(timestamp)

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
