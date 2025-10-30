from loguru import logger
from pathlib import Path
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
import timm
from torch.utils.data import DataLoader

from src.data import Cifar10, Cifar10C
from src.adaptations import CoTTA
from src.utils import ExperimentTracker


# ============================ Setup ================================
@dataclass
class HParams:
    lr: float = 3e-2
    batch_size: int = 128
    n_workers: int = 4
    n_iter: int = 8000
    adapt_alpha: int = 0.99
    adapt_n_augs: int = 5
    adapt_restore_threshold: float = 0.1

dataset, model_name = "cifar10", "vit_base_patch16_224.orig_in21k_ft_in1k"
@dataclass
class Cfg:
    model_name: str = model_name
    dataset: str = dataset
    weights_path: Path = Path("models") / dataset / (model_name + ".pt")
    reset_each_shift: bool = True


cfg, hparams = Cfg(), HParams()

uid = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
exp_name = f"cifar10c_vit_cotta"
exp_path = Path(f"logs/exp_{uid}_{exp_name}")
exp_path.mkdir(parents=True)
et = ExperimentTracker(exp_path, cfg=vars(cfg), hparams=vars(hparams))
logger.add(exp_path / "main.log")
logger.info("Experiment: " + exp_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================== Fine-tune Funciton ===========================
def fineTune(dataset, model, weights_path: Path, n_iter=hparams.n_iter, batch_size=hparams.batch_size):
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        data = DataLoader(dataset, batch_size=batch_size, num_workers=hparams.n_workers)
        upsample = nn.Upsample((224, 224), mode="nearest")
        optimizer = optim.SGD(model.parameters(), lr=hparams.lr, momentum=0.9)

        # FineTune specific layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.head.parameters():
        #     param.requires_grad = True

        loss_sum, i = 0, 0
        logger.info(f"Train on {type(dataset).__name__} for {n_iter} iterations with batch size {batch_size}")
        while(i < n_iter):
            for images, labels in data:
                logits = model(upsample(images).to(device))
                loss = nn.CrossEntropyLoss()(logits, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss
                i += 1
                et.track(train_loss=loss.detach().cpu().numpy())
                if (i + 1) % 100 == 0:
                    loss_sum /= 100
                    logger.info(f"Iterations: [{i + 1}:{n_iter}] -> loss: {loss_sum:.4f}")
                    loss_sum = 0
                if i == n_iter: break

        torch.save(model.state_dict(), weights_path)

# =================== Evaluate Function =============================
def eval(dataset, model, batch_size=hparams.batch_size):
    # data = dataset.load(batch_size=batch_size)
    data = DataLoader(dataset, batch_size=batch_size, num_workers=hparams.n_workers)
    upsample = nn.Upsample((224, 224), mode="nearest")

    with torch.no_grad():
        correct = 0
        start_time = time.time()
        for images, labels in data:
            preds = model(upsample(images).to(device)).argmax(1)
            correct += (preds == labels.to(device)).float().sum()
        time_taken = time.time() - start_time
        acc = correct / len(dataset)
    return acc.cpu().numpy(), time_taken

# ========================= Main Function ===============================
@logger.catch
def main():
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=10).to(device)
    fineTune(Cifar10(), model, cfg.weights_path)

    # ------------ evaluate model -----------------
    test_data = Cifar10(test=True)
    acc, time_taken = eval(test_data, model)
    logger.info(f'Accuracy: {acc:.2%}, Time taken: {time_taken:.3f} s')
    logger.info(f'Samples: {len(test_data)}')
    err = 1. - acc
    et.log({ "acc_source": acc, "tune_time_source": time_taken })
    logger.info(f'Source Error: {err:.2%}')

    # ------------ Setup Adaptation ---------------
    # Adaptation Model
    # features: reset_each_shift
    # ROID / COTTA
    logger.info(f'Adaptation: CoTTA')
    model = CoTTA(model, hparams, device)

    # Corruptions / Domains
    corruptions = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ]

    # severity, if gradual (only in corruption datasets)
    # severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    # else
    severities = [1, 2, 3, 4, 5]
    errs = []

    # workers -> os.cpu_count()
    # ------------ Evaluate on Target ---------------
    for corruption in corruptions:
        for severity in severities:
            dataset = Cifar10C(corruption=corruption, severity=severity)
            logger.info(f"Evaluate on Cifar10C - {corruption} - {severity}")
            acc, time_taken = eval(dataset, model, batch_size=10)
            logger.info(f'Accuracy: {acc:.2%}, Time taken: {time_taken:.3f} s')
            # logger.info(f'Parameter Count: {params / 1e6:.2f} M, GFLOPs: {flops / 1e9:.2f}')
            logger.info(f'Samples: {len(dataset)}')
            et.log({ f"acc_{corruption}_{severity}": acc, f"eval_time_{corruption}_{severity}": time_taken })
            err = 1. - acc
            errs.append(err)
            logger.info(f'{corruption.upper()}_{severity} error: {err:.2%}')
    logger.info(f'Mean Error: {np.mean(errs):.2%}')
    et.log({ "acc_mean": 1. - np.mean(errs)})

    # ------------ Save adapted model ---------------
    # Save the model
    # import copy
    # best_model_weights = copy.deepcopy(model.state_dict())
    # torch.save(best_model_weights, f"{corruption}_{severities[0]}_{model_arch}.pth")


if __name__ == "__main__":
    main()
