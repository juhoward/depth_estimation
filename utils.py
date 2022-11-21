from math import sqrt, dist
from statistics import median, mean


class res_dict(dict):
    def __init__(self):
        self = dict()
    def add(self, key, value):
        self[key] = [value]

class Results(object):
    def __init__(self) -> None:
        self.results = res_dict()
        self.history = res_dict()
    def rmse(self, yhat, y, method_name):
        metric_name = method_name + '_rmse'
        errors = list(map(lambda x: (x[0] - x[1])**2, zip(yhat, y)))
        error = mean(errors)
        self.results[metric_name] = error
        self.history[method_name] = errors

    def mae(self, yhat, y, method_name):
        metric_name = method_name + '_rmse'
        errors = list(map(lambda x: abs(x[0] - x[1]), zip(yhat, y)))
        error = mean(errors)
        self.results[metric_name] = error
        self.history[method_name] = errors
    
    def report(self):
        for k, v in self.results.items():
            print(f'\n{k}: {v}\n')

    def update(self, dist, method_name):
        if method_name not in self.history.keys():
            self.history.add(method_name, dist)
        else:
            self.history[method_name].append(dist)
