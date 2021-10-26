from experiments.image_retrieval.tokyo247 import TokyoRetrievalAgent
from experiments.image_retrieval.radenovic_data import RadenovicRetrievalAgent


class ImageRetrievalFactory(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_retrieval_agent(self):
        agent = None
        dataset_name = self.cfg.task.task_params.dataset
        if dataset_name == 'tokyo247':
            agent = TokyoRetrievalAgent(self.cfg)
        elif dataset_name in ['roxford5k', 'rparis6k', 'oxford5k', 'paris6k']:
            agent = RadenovicRetrievalAgent(self.cfg)

        return agent
