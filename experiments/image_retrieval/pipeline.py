from experiments.service.benchmark_base import Benchmark
from experiments.service.image_retrieval_factory import ImageRetrievalFactory


class ImageRetrievalBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        # create agent
        self.retrieval_agent = ImageRetrievalFactory(
            self.cfg
        ).get_retrieval_agent()

    def evaluate(self):
        self.retrieval_agent.retrieve()
