import collections
from pandas import DataFrame


class Meter:
    def __init__(self, key_spec):
        self.key_spec = key_spec

    def compute(self, model, metadata, device):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatMeter(self, other)

    def __call__(self, model, metadata, device):
        values = self.compute(model, metadata, device)
        result = collections.OrderedDict()
        for key, value in zip(self.key_spec, values):
            value = [value] if not type(value) is list else value
            result[key] = value
        result.update({key: [metadata[key]] * len(value) for key in metadata})
        result = DataFrame.from_dict(result)
        return result


class ConcatMeter(Meter):
    def __init__(self, car, cdr):
        super().__init__(car.key_spec + cdr.key_spec)
        self.car = car
        self.cdr = cdr

    def compute(self, model, metadata, device):
        return self.car.compute(model, metadata) + self.cdr.compute(model, metadata)
