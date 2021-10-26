import numpy as np
from os import path as osp
import pickle
from experiments.image_retrieval.base import BaseRetrievalAgent
from experiments.image_retrieval.download_radenovic_data import download_datasets
from experiments.image_retrieval.utils import compute_map_and_print


class RadenovicRetrievalAgent(BaseRetrievalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.available_datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
        self.data = self._create_retrieval_data()

    @staticmethod
    def config_imname(cfg, i):
        return osp.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

    @staticmethod
    def config_qimname(cfg, i):
        return osp.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

    def _config_dataset(self, dir_main, dataset):
        dataset = dataset.lower()

        if dataset not in self.available_datasets:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = osp.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname

        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = osp.join(dir_main, dataset)
        cfg['dir_images'] = osp.join(cfg['dir_data'], 'jpg')

        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])

        cfg['im_fname'] = RadenovicRetrievalAgent.config_imname
        cfg['qim_fname'] = RadenovicRetrievalAgent.config_qimname

        cfg['dataset'] = dataset
        return cfg

    def _create_retrieval_data(self):
        cfg = self._config_dataset(self.cfg.paths.datasets_home_dir,
                                   self.retrieval_agent_cfg.dataset)
        q_images = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        db_images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        return {'qs': q_images,
                'dbs': db_images,
                'bbxs': bbxs}

    def retrieve(self):
        print(f'Downloading the revisited datasets (roxford5k, rparis6k)...')
        download_datasets(self.cfg.paths.datasets_home_dir)
        print(f'Downloading the revisited datasets (roxford5k, rparis6k)... Done!')

        print(f'Extract keypoints and local descriptors from the database images...')
        self.ldd_model.evaluate(self.data['dbs'])

        print(f'Extract keypoints and local descriptors from the query images...')
        self.ldd_model.evaluate(self.data['qs'], bbxs=self.data['bbxs'])

        # Load GND data
        with open(osp.join(self.retrieval_agent_cfg.paths.img_home_dir,
                           f'gnd_{self.retrieval_agent_cfg.dataset}.pkl'), "rb") as f:
            cfg_gnd = pickle.load(f)

        # Let us load Radenovic image retrieval resutls
        with open(self.retrieval_agent_cfg.paths.radenovic_dict, "rb") as f:
            data_rad = pickle.load(f)

        db_rank_orig_dict = dict(zip(data_rad["db_images"], range(len(data_rad["db_images"]))))
        new_ranks = np.zeros(data_rad["ranks"].shape)
        db_feats_dict = {}
        for i_q, query in enumerate(data_rad["q_images"]):
            ranks_rad = data_rad["ranks"][:, i_q]
            db_rerank_rad = np.array(data_rad["db_images"])[ranks_rad]
            orig_top_k_fnames = db_rerank_rad[:self.retrieval_agent_cfg.topk]

            pkl_fname = query.split('/')[-1][:-3] + 'pkl'
            with open(osp.join(self.retrieval_agent_cfg.output.precomputed_feats_dir, 'jpg', pkl_fname), 'rb') as f:
                q_kpt_data = pickle.load(f)

            inls_count = []
            for db_fname in orig_top_k_fnames:
                pkl_fname = db_fname.split('/')[-1][:-3] + 'pkl'
                if pkl_fname not in db_feats_dict:
                    with open(osp.join(self.retrieval_agent_cfg.output.precomputed_feats_dir, 'jpg', pkl_fname), 'rb') as f:
                        d_kpt_data = pickle.load(f)
                    db_feats_dict[pkl_fname] = d_kpt_data
                else:
                    d_kpt_data = db_feats_dict[pkl_fname]

                n_inliers, _, _ = self.matcher.get_inliers_count(q_kpt_data, d_kpt_data)
                inls_count.append(n_inliers)
            rerank_top_k = [orig_top_k_fnames[idx] for idx in (-np.asarray(inls_count)).argsort()]

            for item in db_rerank_rad[self.retrieval_agent_cfg.topk:]:
                rerank_top_k.append(item)

            rerank_full = rerank_top_k
            for i_d, db_rerank in enumerate(rerank_full):
                new_ranks[i_d, i_q] = db_rank_orig_dict[db_rerank]

            print("Query ", i_q, " out of ", len(data_rad["q_images"]), " processed")

        output = compute_map_and_print(self.retrieval_agent_cfg.dataset,
                                       new_ranks,
                                       cfg_gnd["gnd"])

        # write results to the file
        with open(self.cfg.task.task_params.output.res_fname, "w") as f:
            f.write(f"mAP benchmark:\n")
            mapE, mapM, mapH = output['mAP']
            f.write(f'E: {mapE*100:04.2f}, M: {mapM*100:04.2f}, H: {mapH*100:04.2f}\n')
            f.write(f"mP@k [1, 5, 10] benchmark:\n")
            mprE, mprM, mprH = output['mP@k']
            for type_, res in zip(['E', 'M', 'H'], [mprE, mprM, mprH]):
                f.write(f'{type_}: [{res[0]*100:04.2f}, {res[1]*100:04.2f}, {res[2]*100:04.2f}]\n')

        print('Done')
