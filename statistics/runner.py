import glob
from queue import Queue
from threading import Thread

import pandas as pd
import torch
from tqdm import tqdm

import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 

from training.model import BaselineModel


class ParallelExperimentRunner:
    def __init__(self, root, file_parse, meter, num_workers, out, model_class=BaselineModel, devices=None, random=False):
        if devices is None:
            devices = ['cpu', 'cuda:0']
        model_list = glob.glob(os.path.join(root, '*.pt'))
        
        self.model_queue = Queue()
        for model in model_list:
            self.model_queue.put((model, file_parse(model)))
        self.len = len(model_list)
        self.meter = meter
        self.num_workers = num_workers
        self.out = out
        self.devices = devices
        self.random = random
        self.model_class = model_class

    def make_worker(self, progress_bar, sink, device):
        def worker():
            while True:
                if self.model_queue.empty():
                    break
                model_file, metadata = self.model_queue.get()
                model = self.model_class(metadata['n_bn'], metadata['d_vvs'], metadata['n_ch']).to(device)

                if not self.random:
                    state_dict = torch.load(model_file, map_location=device)
                    try:
                        model.load_conv_dict(state_dict)
                    except:
                        model.load_state_dict(state_dict)
                    # torch.save(model.conv_dict(), model_file)
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

    runner = ParallelExperimentRunner('../models/colour-mos', file_parse, SpatialOpponency(), 0, 'spatial-mos.pd', model_class=BaselineModel, devices=['cuda']) #0 to debug
    runner.run()
