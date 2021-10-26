import os
from os import path as osp
import hydra
import pickle
from tqdm import tqdm
import torch
from data.dataset import MegaDepthDataset
from data.augmentations import get_img_augmentations


@hydra.main(config_path="configs", config_name="gen_val_dataset")
def main(cfg):
    data_params = cfg.data_params
    set_seed(cfg.seed)

    if not osp.exists(cfg.output_params.output_path):
        os.makedirs(cfg.output_params.output_path)

    _, val_img_augs = get_img_augmentations()
    # create dataset
    dataset = MegaDepthDataset(img_path=osp.join(data_params.img_dir, 'test'),
                               kpts_path=data_params.kpts_descs_dir,
                               global_desc_path=data_params.global_descs_dir,
                               crop_size=data_params.crop_size,
                               transforms=val_img_augs)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=cfg.n_workers)

    for i, mini_batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for key in list(mini_batch.keys()):
            if key == 'img_fname':
                continue
            mini_batch[key] = mini_batch[key].squeeze(0)

        with open(osp.join(cfg.output_params.output_path, f'meta_data_{i}.pkl'), 'wb') as f:
            pickle.dump(mini_batch, f)


if __name__ == '__main__':
    main()