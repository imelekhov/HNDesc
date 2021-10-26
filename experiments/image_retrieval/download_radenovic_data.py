import os


def download_datasets(datasets_dir):
    """
    DOWNLOAD_DATASETS Checks, and, if required, downloads the necessary datasets for the testing.

        download_datasets(DATA_ROOT) checks if the data necessary for running the example script exist.
        If not it downloads it in the folder structure:
            DATA_ROOT/datasets/roxford5k/ : folder with Oxford images
            DATA_ROOT/datasets/rparis6k/  : folder with Paris images
    """

    # Create data folder if it does not exist
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)

    # Download datasets folders datasets/DATASETNAME/
    datasets = ['roxford5k', 'rparis6k']
    for di in range(len(datasets)):
        dataset = datasets[di]

        if dataset == 'roxford5k':
            src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images.tgz']
        elif dataset == 'rparis6k':
            src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1.tgz', 'paris_2.tgz']
        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        dst_dir = os.path.join(datasets_dir, dataset, 'jpg')
        if not os.path.isdir(dst_dir):
            print('>> Dataset {} directory does not exist. Creating: {}'.format(dataset, dst_dir))
            os.makedirs(dst_dir)
            for dli in range(len(dl_files)):
                dl_file = dl_files[dli]
                src_file = os.path.join(src_dir, dl_file)
                dst_file = os.path.join(dst_dir, dl_file)
                print('>> Downloading dataset {} archive {}...'.format(dataset, dl_file))
                os.system('wget {} -O {}'.format(src_file, dst_file))
                print('>> Extracting dataset {} archive {}...'.format(dataset, dl_file))
                # create tmp folder
                dst_dir_tmp = os.path.join(dst_dir, 'tmp')
                os.system('mkdir {}'.format(dst_dir_tmp))
                # extract in tmp folder
                os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))
                # remove all (possible) subfolders by moving only files in dst_dir
                os.system('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
                # remove tmp folder
                os.system('rm -rf {}'.format(dst_dir_tmp))
                print('>> Extracted, deleting dataset {} archive {}...'.format(dataset, dl_file))
                os.system('rm {}'.format(dst_file))

        gnd_src_dir = os.path.join('http://cmp.felk.cvut.cz/revisitop/data', 'datasets', dataset)
        gnd_dst_dir = os.path.join(datasets_dir, dataset)
        gnd_dl_file = 'gnd_{}.pkl'.format(dataset)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
        if not os.path.exists(gnd_dst_file):
            print('>> Downloading dataset {} ground truth file...'.format(dataset))
            os.system('wget {} -O {}'.format(gnd_src_file, gnd_dst_file))
