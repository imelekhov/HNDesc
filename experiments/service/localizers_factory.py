from experiments.localization.aachen import AachenLocalizer


class LocalizersFactory(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_localizer(self):
        localizer = None
        loc_name = self.cfg.task.task_params.dataset
        if loc_name == 'aachen_v11':
            localizer = AachenLocalizer(self.cfg)

        return localizer
