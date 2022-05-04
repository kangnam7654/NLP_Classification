import torch


def opt_load(config, model):
    optimizer = None
    if config["MODEL"]["OPTIMIZER"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["MODEL"]["LR"],
            weight_decay=config["MODEL"]["WEIGHT_DECAY"],
        )
    if config["MODEL"]["OPTIMIZER"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["MODEL"]["LR"],
            weight_decay=config["MODEL"]["WEIGHT_DECAY"],
        )
    if config["MODEL"]["OPTIMIZER"] == "SGD-Nesterov":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["MODEL"]["LR"],
            weight_decay=config["MODEL"]["WEIGHT_DECAY"],
            momentum=config["MODEL"]["MOMENTUM"],
            nesterov=True,
        )
    return optimizer
