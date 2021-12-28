import hydra
from pathlib import Path
from experiments.hpatches.pipeline import HPSequenceBenchmark

"""
from experiments.localization.pipeline import VisualLocBenchmark
from experiments.image_retrieval.pipeline import ImageRetrievalBenchmark
from experiments.image_matching.pipeline import ImageMatchingBenchmark
"""


@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    for name in cfg["paths"]:
        if name == "project_dir":
            cfg["paths"]["project_dir"] = str(
                Path(hydra.utils.get_original_cwd()).parents[0]
            )

    benchmark = None
    if cfg.task.task_params.name == "hpatches":
        benchmark = HPSequenceBenchmark(cfg)
    """
    elif cfg.task.task_params.name == 'localization':
        benchmark = VisualLocBenchmark(cfg)
    elif cfg.task.task_params.name == 'image_retrieval':
        benchmark = ImageRetrievalBenchmark(cfg)
    elif cfg.task.task_params.name == 'image_matching':
        benchmark = ImageMatchingBenchmark(cfg)
    """

    benchmark.evaluate()


if __name__ == "__main__":
    main()
