#NOTE: this file depends on generate_initial_pop.py
import numpy as np

from .data_collector import DataCollector

from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import Hypervolume

class Performance_Indicators(DataCollector):

    def __init__(self, 
                calc_gd=True, 
                calc_igd=True, 
                calc_gd_plus=True, 
                calc_igd_plus=True, 
                hv_ref_points=[],
                normalize_performance_indicators=True) -> None:

        self.filename ="performance_indicators"
        self.normalize_performance_indicators = normalize_performance_indicators
        data_keys = ["generation", "eval"]
        if calc_gd:
            data_keys.append("gd")
        if calc_igd:
            data_keys.append("igd")
        if calc_gd_plus:
            data_keys.append("gd+")
        if calc_igd_plus:
            data_keys.append("igd+")
        if len(hv_ref_points) == 1:
            data_keys.append("hv")
        elif len(hv_ref_points) < 1:
            for i in range(len(hv_ref_points)):
                data_keys.append("hv_p"+str(i))
        
        super().__init__(data_keys=data_keys, filename="performance_indicators")

    def notify(self, algorithm):
        generation = algorithm.n_gen
        evals = algorithm.evaluator.n_eval
        _F = algorithm.opt.get('F')
        pareto_front = algorithm.problem.pareto_front()

        for key in self.data.keys():
            if key == "generation":
                self.data[key].append(generation)
            elif key == "eval":
                self.data[key].append(evals)
            elif key == "gd":
                self.data[key].append( GD(pf=pareto_front, zero_to_one=self.normalize_performance_indicators).do(_F) )
            elif key == "igd":
                self.data[key].append( IGD(pf=pareto_front, zero_to_one=self.normalize_performance_indicators).do(_F) )
            elif key == "gd+":
                self.data[key].append( GDPlus(pf=pareto_front, zero_to_one=self.normalize_performance_indicators).do(_F) )
            elif key == "igd+":
                self.data[key].append( IGDPlus(pf=pareto_front, zero_to_one=self.normalize_performance_indicators).do(_F) )
            elif key[:2] == "hv":
                point_index = int(key.split("_")[1])
                self.data[key].append( Hypervolume(ref_point=hv_ref_points[point_index], zero_to_one=self.normalize_performance_indicators).do(_F) )