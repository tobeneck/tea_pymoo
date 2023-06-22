import pandas as pd

from os.path import exists
from pathlib import Path

from pymoo.core.callback import Callback

class AccumulateCallbacks(Callback):

    def __init__(self, collectors):
        '''
        this class acts as a wrapper for a list of callbacks that is called every generation

        Parameters
        ----------
        collectors : list of DataCollectors
            a list of data collectors that is called every generation
        '''
        super().__init__()
        self.collectors = collectors
    
    def notify(self, algorithm):
        for collector in self.collectors:
            collector.notify(algorithm)
    
    def finalize(self, out_path):
        '''
        call this at the end of a run to get the data of each collector to save the data from the collector to the out_path

        Parameters
        ----------
        out_path : string
            the filepath the data should be saved to
        '''
        for collector in self.collectors:
            df = pd.DataFrame(collector.data)
            output_filename = Path(out_path) / str(collector.filename+".csv")
            df.to_csv(output_filename, mode='a', index=False, header=(not exists(output_filename)) )

            #output_filename = Path(out_path) / str(collector.filename+".csv.gz")
            #df.to_csv(output_filename, mode='a', index=False, header=(not exists(output_filename)), compression="gzip")