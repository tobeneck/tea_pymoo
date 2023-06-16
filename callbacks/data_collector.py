from pymoo.core.callback import Callback


class DataCollector(Callback):
    def __init__(self, data_keys, filename):
        super().__init__()
        self.data = { key : [] for key in data_keys}
        self.filename = filename