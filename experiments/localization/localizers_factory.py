from experiments.localization.aachen import AachenLocalizer

VALID_SCENES = ["aachen_v11_night"]


def localizers_factory(cfg):
    scene_name = cfg.scene.name
    if scene_name not in VALID_SCENES:
        raise ValueError("Given scene is not valid.")

    localizer = None
    if scene_name == "aachen_v11_night":
        localizer = AachenLocalizer(cfg)

    return localizer
