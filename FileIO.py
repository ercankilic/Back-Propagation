import random as rd

class FileIO:
    def __init__(self, path, delimiter=",", encoding="utf8"):
        self.path = path
        self.delimiter = delimiter
        self.encoding = encoding

    def read_file(self):
        with open(self.path, mode="r", encoding=self.encoding) as file:
            self.data = []
            lines = file.readlines()
            rd.shuffle(lines)
            for line in lines:
                line = line.replace("\n", "")
                line = line.split(self.delimiter)
                while '' in line:
                    line.remove('')
                self.data.append(line)

    def write_file(self, content):
        pass