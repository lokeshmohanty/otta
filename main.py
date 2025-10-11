from loguru import logger
from pathlib import Path
import time
import datetime
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
import timm
from torch.utils.data import DataLoader

from src.data import Cifar10, Cifar10C
from src.utils import ExperimentTracker

class HParams:
    lr: float = 1e-3
    batch_size: int = 100
    n_workers: int = 4
    n_iter: int = 800

model_name = "vit_base_patch16_224.orig_in21k_ft_in1k"
class Cfg:
    model_name = model_name
    weights_path = Path("models/cifar10") / model_name

uid = datetime.datetime.now()
et = ExperimentTracker(f"CIFAR10C_vit_source_{uid}", cfg=Cfg, hparams=HParams)

logger.add(f"logs/main_{uid}.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataset, model, weights_path: Path, n_iter=HParams.n_iter, batch_size=HParams.batch_size):
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        return
    # else:
    data = DataLoader(dataset, batch_size=batch_size, num_workers=HParams.n_workers)
    upsample = nn.Upsample((224, 224), mode="nearest")
    optimizer = optim.SGD(model.parameters(), lr=HParams.lr, momentum=0.9)

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.head.parameters():
    #     param.requires_grad = True

    loss_sum = 0
    logger.info(f"Train on {type(dataset).__name__} for {n_iter} iterations with batch size {batch_size}")
    for i, (images, labels) in enumerate(data):
        logits = model(upsample(images).to(device))
        loss = nn.CrossEntropyLoss()(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        et.track(train_loss=loss)
        if i == n_iter: break
        if (i + 1) % 100 == 0:
            loss_sum /= 100
            logger.info(f"Iterations: [{i + 1}:{n_iter}] -> loss: {loss_sum:.4f}")
            loss_sum = 0

    torch.save(model.state_dict(), weights_path)


def eval(dataset, model, batch_size=HParams.batch_size):
    data = dataset.load(batch_size=batch_size)
    # data = DataLoader(dataset, batch_size=200, num_workers=8)
    upsample = nn.Upsample((224, 224), mode="nearest")

    with torch.no_grad():
        correct = 0
        start_time = time.time()
        for images, labels in data:
            preds = model(upsample(images).to(device)).argmax(1)
            correct += (preds == labels.to(device)).float().sum()
        time_taken = time.time() - start_time
        acc = correct / len(dataset)
    return acc.cpu(), time_taken

def main():
    model = timm.create_model(Cfg.model_name, pretrained=True, num_classes=10).to(device)

    # FineTune on Cifar10
    train(Cifar10(), model, Cfg.weights_path)

    test_data = Cifar10(test=True)
    acc, time_taken = eval(test_data, model)
    logger.info(f'Accuracy: {acc:.2%}, Time taken: {time_taken:.3f} s')
    logger.info(f'Samples: {len(test_data)}')
    err = 1. - acc
    logger.info(f'Source Error: {err:.2%}')


    # Adaptation Model
    # features: reset_each_shift
    # ROID / COTTA

    # Corruptions / Domains
    corruptions = ["fog", "brightness"]

    # severity, if gradual (only in corruption datasets)
    # severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    # else
    severities = [5]
    errs = []

    # workers -> os.cpu_count()
    for corruption in corruptions:
        dataset = Cifar10C(corruption=corruption, severity=severities[0])
        acc, time_taken = eval(dataset, model)

        logger.info(f"Evaluate on Cifar10C - {corruption}")
        logger.info(f'Accuracy: {acc:.2%}, Time taken: {time_taken:.3f} s')
        # logger.info(f'Parameter Count: {params / 1e6:.2f} M, GFLOPs: {flops / 1e9:.2f}')
        logger.info(f'Samples: {len(dataset)}')
        et.metrics[f"accuracy_{corruption}"] = acc
        et.metrics[f"eval_time_{corruption}"] = time_taken
        err = 1. - acc
        errs.append(err)
        logger.info(f'{corruption.upper()} error: {err:.2%}')
    logger.info(f'Mean Error: {np.mean(errs):.2%}')



    # Save the model
    # import copy
    # best_model_weights = copy.deepcopy(model.state_dict())
    # torch.save(best_model_weights, f"{corruption}_{severities[0]}_{model_arch}.pth")


if __name__ == "__main__":
    main()
