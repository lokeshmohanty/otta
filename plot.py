from pathlib import Path
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import tyro

from src.data import Cifar10, Cifar100

sns.set_theme()

def visualize_model():
    model_graph = draw_graph(model, input_size=(1, 3, 224, 224), device='meta')
    model_graph.visualize_model

def visualize_data():
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
    ]
    datasets = [CIFAR10C(corruption=corr) for corr in corruptions] 
    fig, axes = plt.subplots(3, 5, figsize=(8,6))
    for i in range(3):
        for j in range(5):
            ax = axes[i][j]
            ind = 3 * i + j
            ax.set_title(corruptions[ind])
            ax.axis('off')
            ax.imshow(datasets[ind].images[11].permute(1, 2, 0))

    plt.savefig("cifar10c.svg")

def main(folder: Path, /, severity: int = 0) -> None:
    """Generate plots

    Args:
        file: path to the file
    """
    file = Path(folder) / 'data.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    metrics = data["metrics"]
    avg_eval_time = np.array([v for k, v in metrics.items() if k.split('_')[0] == 'eval']).mean()

    pprint(f"Data file: {file}")
    pprint(f"Config: {data['cfg']}")
    pprint(f"Hyperparameters: {data['hparams']}")
    pprint(f"Mean time taken per iteration(1 batch): {avg_eval_time:.3f} s")

    if severity:
        err = [(k[4:], 1. - v) for k, v in metrics.items() if k.split('_')[0] == 'acc'][1:]
        s = []
        for i in range(1, 6):
            items = [(k[:-2], v) for k, v in err if k.split('_')[-1] == str(i)]
            s.append(([k for k, _ in items], [v.item() for _, v in items])) 

        df = pd.DataFrame(data={
            'corruption': s[0][0],
            'acc.1': s[0][1],
            'acc.2': s[1][1],
            'acc.3': s[2][1],
            'acc.4': s[3][1],
            'acc.5': s[4][1],
        })

        # Error bar plot
        plt.bar(*s[severity - 1])
        plt.xticks(rotation=90)
        plt.yticks(np.arange(0, 1, 0.1))
        plt.ylabel("error")
        plt.savefig(file.parent / f"err_{severity}.png")

    else:
        err = [(k[4:], 1. - v) for k, v in metrics.items() if k.split('_')[0] == 'acc' and not k.split('_')[-1].isdigit()][1:]

        # Error bar plot
        plt.bar([k for k, _ in err], [v for _, v in err])
        plt.xticks(rotation=90)
        plt.yticks(np.arange(0, 1, 0.1))
        plt.ylabel("error")
        plt.savefig(file.parent / f"err_mean.png")

    if data["tracks"]:
        for k, v in data["tracks"].items():
            plt.plot(v)
            plt.ylabel(k)
            plt.xlabel("iterations")
            plt.savefig(file.parent / f"{k}.png")


if __name__ == "__main__":
    tyro.cli(main)
