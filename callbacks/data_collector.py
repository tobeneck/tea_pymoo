from pymoo.core.callback import Callback


class DataCollector(Callback):
    def __init__(self, data_keys, filename, additional_run_info=None):
        super().__init__()

        keys = data_keys

        self.additional_run_info = additional_run_info
        if self.additional_run_info:
            for key in self.additional_run_info.keys():
                keys.append(key)

        self.data = { key : [] for key in keys}
        
        self.filename = filename

    def handle_additional_run_info(self):
        if self.additional_run_info:
            for key in self.additional_run_info.keys():
                self.data[key].append(self.additional_run_info[key])