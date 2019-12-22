import os
from pathlib import Path
from time import sleep

from BaseAPI import BaseAPI

class LSTM_API(BaseAPI):
    def __init__(self):
        self.path = Path(__file__)
        super(LSTM_API, self).__init__()

    def run_example(self, text: str):
        command = f"""python {self.path.parent / "LSTM.py"} --text="{text}" """
        result_file: Path = self.path.parent / "result.txt"
        if result_file.exists():
            os.remove(result_file)
        os.system(command=command)
        wait_times = 0
        while not result_file.exists() and wait_times<20:
            wait_times += 1
            sleep(1)
        if wait_times>=20:
            result = -1
        else:
            with open(result_file) as f:
                result = int(f.read())
        if result_file.exists():
            os.remove(result_file)
        return result

if __name__ == "__main__":
    text = "热烈庆祝中华人民共和国成立"
    api = LSTM_API()
    print(api.run_example(text))