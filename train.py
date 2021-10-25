import hydra
from hndesc.pipeline import Pipeline


@hydra.main(config_path="configs", config_name="main_m_and_p_sh")
def main(cfg):
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
