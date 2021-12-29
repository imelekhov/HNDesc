import os
from os import path as osp
import shutil
from experiments.service.benchmark_base import Benchmark
from experiments.service.localizers_factory import LocalizersFactory


class VisualLocBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        output_localization_dir = self.cfg.task.task_params.output.loc_res_dir
        if osp.exists(output_localization_dir):
            shutil.rmtree(output_localization_dir)
        os.makedirs(output_localization_dir)

        # create localizer
        self.localizer = LocalizersFactory(self.cfg).get_localizer()

    def evaluate(self):
        self.localizer.localize()
