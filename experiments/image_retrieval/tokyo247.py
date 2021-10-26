import numpy as np
from os import path as osp
import scipy.io as sio
import pickle
from tqdm import tqdm
from experiments.image_retrieval.base import BaseRetrievalAgent


class TokyoRetrievalAgent(BaseRetrievalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.q_db_dict = self._create_q_db_dict()
        self.precomputed_feats_dir = self.retrieval_agent_cfg.output.precomputed_feats_dir
        self.res = {}

    def _create_q_db_dict(self):
        shortlist_knn = sio.loadmat(self.retrieval_agent_cfg.paths.shortlist_dict_fname)
        db_data = shortlist_knn['ImgList'][0, :]

        q_db_dict = {}
        for q_id in range(len(db_data)):
            query = db_data[q_id][0][0]
            q_db_dict[query] = [db_fname[0][:-4] + '.jpg' for db_fname in db_data[q_id][1][0]]
        return q_db_dict

    def retrieve(self):
        # Let us extract keypoints and local descriptors from query and db images first. So, let us create a list of
        # image filenames
        print(f'Create list of images...')
        fnames = []
        for i, q in enumerate(self.q_db_dict):
            q_fname = osp.join(self.retrieval_agent_cfg.paths.q_imgs_dir, q)
            db_fnames = [osp.join(self.retrieval_agent_cfg.paths.db_imgs_dir, db_name) for db_name in self.q_db_dict[q]]
            fnames2add = [q_fname] + db_fnames
            fnames.extend(fnames2add)
        print(f'Create list of images... Done!')

        # Extract features
        self.ldd_model.evaluate(fnames)

        for _, q_fname in enumerate(tqdm(self.q_db_dict, total=len(list(self.q_db_dict.keys())))):
            with open(osp.join(self.precomputed_feats_dir, '247query', q_fname[:-3] + 'pkl'), "rb") as f:
                kpts_descs_q = pickle.load(f)

            self.res[q_fname] = {}
            inls_count_per_q = []
            for db in self.q_db_dict[q_fname]:
                with open(osp.join(self.precomputed_feats_dir, '247_db', db[:-3] + 'pkl'), "rb") as f:
                    kpts_descs_db = pickle.load(f)

                n_inliers, _, _ = self.matcher.get_inliers_count(kpts_descs_q, kpts_descs_db)

                inls_count_per_q.append(n_inliers)

            self.res[q_fname]['inliers'] = inls_count_per_q
            self.res[q_fname]['ranked_db'] = [self.q_db_dict[q_fname][idx]
                                              for idx in (-np.asarray(inls_count_per_q)).argsort()]

        with open(self.retrieval_agent_cfg.output.res_fname, 'wb') as f:
            pickle.dump(self.res, f)
        print('Done')
