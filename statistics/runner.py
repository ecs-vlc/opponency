import glob
from queue import Queue
from threading import Thread

import pandas as pd
import torch
from tqdm import tqdm

import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from training.model import BaselineModel


class WeightInit:
    def initialise(self, target_model: torch.nn.Module, file: str):
        raise NotImplementedError

class PreTrained(WeightInit):
    def initialise(self, target_model: torch.nn.Module, file: str):
        state_dict = torch.load(file, map_location=next(target_model.parameters()).device)
        try:
            target_model.load_conv_dict(state_dict)
        except:
            target_model.load_state_dict(state_dict)
        return target_model


class Random(WeightInit):
    def initialise(self, target_model: torch.nn.Module, file: str):
        return target_model


class Iid(WeightInit):
    def __init__(self, refrence_model):
        self.reference_model = refrence_model

    def initialise(self, target_model: torch.nn.Module, file: str):
        for random_layer, trained_layer in zip(target_model.retina + target_model.ventral, self.reference_model.retina + self.reference_model.ventral):
            layer_name, random_layer = random_layer
            _, trained_layer = trained_layer
            if 'conv' in layer_name:
                random_layer.weight.data.normal_(mean=trained_layer.weight.mean().item(),
                                                 std=trained_layer.weight.std().item())
                if trained_layer.bias.size(0) > 1:
                    random_layer.bias.data.normal_(mean=trained_layer.bias.mean().item(),
                                                   std=trained_layer.bias.std().item())
                else:
                    random_layer.bias.data.fill_(trained_layer.bias.item())
        return target_model


class ParallelExperimentRunner:
    def __init__(self, root, file_parse, meter, num_workers, out, model_class=BaselineModel, devices=None, weight_init: WeightInit = PreTrained()):
        if devices is None:
            devices = ['cpu', 'cuda:0']
        model_list = glob.glob(os.path.join(root, '*.pt'))
        
        self.model_queue = Queue()
        for model in model_list:
            self.model_queue.put((model, file_parse(model.split('/')[-1])))
        self.len = len(model_list)
        self.meter = meter
        self.num_workers = num_workers
        self.out = out
        self.devices = devices
        self.weight_init = weight_init
        self.model_class = model_class

    def make_worker(self, progress_bar, sink, device):
        def worker():
            while True:
                if self.model_queue.empty():
                    break
                model_file, metadata = self.model_queue.get()
                model = self.model_class(metadata['n_bn'], metadata['d_vvs'], metadata['n_ch']).to(device)

                self.weight_init.initialise(model, model_file)

                for param in model.parameters():
                    param.requires_grad = False
                res = self.meter(model, metadata, device)
                sink.put(res)
                self.model_queue.task_done()
                progress_bar.update()
        return worker

    def make_aggregator(self, sink):
        def worker():
            while True:
                result = sink.get()
                worker.frames.append(result)
                sink.task_done()
        worker.frames = []
        return worker

    def run(self):
        bar = tqdm(total=self.len)

        sink = Queue()
        for i in range(self.num_workers):
            t = Thread(target=self.make_worker(bar, sink, self.devices[i % len(self.devices)]))
            t.daemon = True
            t.start()
        if self.num_workers == 0:
            self.make_worker(bar, sink, self.devices[0 % len(self.devices)])()

        aggregator = self.make_aggregator(sink)
        t = Thread(target=aggregator)
        t.daemon = True
        t.start()

        self.model_queue.join()
        sink.join()
        frame = pd.concat(aggregator.frames, ignore_index=True)
        frame.to_pickle(self.out)


if __name__ == "__main__":
    # from rfdeviation import RFDeviation
    from statistics.devalois import DeValois
    from statistics.spatial_opponency import SpatialOpponency
    # from orientation import RFOrientation
    from training import BaselineModel
    from training.model_imagenet import ImageNetModel

    def file_parse(file):
        v = file.split('.')[0].split('_')
        return {'n_bn': int(v[1]), 'd_vvs': int(v[2]), 'rep': int(v[3]), 'n_ch': 3}

    reference_model = BaselineModel(32, 4, 3)
    reference_model = PreTrained().initialise(reference_model, '../../models/colour/model_32_4_0.pt')

    runner = ParallelExperimentRunner('../../models/colour', file_parse, SpatialOpponency(), 0, 'spatial-iid.pd', model_class=BaselineModel, devices=['cuda'], weight_init=Iid(reference_model)) #0 to debug
    runner.run()
