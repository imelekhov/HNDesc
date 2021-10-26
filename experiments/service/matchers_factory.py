from assets.matchers.knn_matcher import KnnMatcher


class MatchersFactory(object):
    def __init__(self, matcher_cfg):
        self.matcher_cfg = matcher_cfg

    def get_matcher(self):
        matcher = None
        matcher_name = self.matcher_cfg.matcher_params.name
        if matcher_name == 'knn':
            matcher = KnnMatcher(self.matcher_cfg)

        return matcher
