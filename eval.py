# Third party import
from argparse import ArgumentParser
import yaml
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib

# Local import
from kd4jets.models import MLPKD, DeepSetKD
from kd4jets import boost_jets, boost_batch

matplotlib.use("agg")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--cfg", type = str, required = True)
    args = parser.parse_args()
    return args

def tocuda(x):
    if isinstance(x, list):
        return [y.cuda() for y in x]
    else:
        return x.cuda()
        
def plot_roc(name, df, ax):
    new_df = df[df["beta"] == 0]
    fpr, tpr, _ = roc_curve(np.array(new_df["labels"]), np.array(new_df["prob"]))
    fpr, tpr = fpr[(tpr > 0) & (fpr > 0)], tpr[(tpr > 0) & (fpr > 0)]
    ax.plot(tpr, 1/fpr, label = name, ms = 1)

def plot_boost(name, df, ax):
    df["acc"] = (df["pred"] == df["labels"])
    acc = df.groupby("beta").mean()
    ax.plot(acc.index, acc["acc"], ms = 1, ls="-", label=name)
    
@torch.no_grad()
def get_predictions(model):
    loader = model.test_dataloader()
    dfs = []
    model.eval()
    for batch in tqdm(loader):
        for name in batch:
            batch[name] = tocuda(batch[name])
        for beta in np.linspace(0, 1, 20, endpoint=False):
            batched_beta = torch.tensor([beta]*model.hparams["batch_size"])
            logit, latent_rep = model.student(boost_batch(batch, batched_beta))
            new_df = pd.DataFrame({
                "beta": beta,
                "pred": logit.cpu().argmax(-1).numpy(),
                "prob": torch.sigmoid(logit[:, 1] - logit[:, 0]).cpu().numpy(),
                "labels": batch["is_signal"].cpu().float().numpy(),
            })
            dfs.append(new_df)
    df = pd.concat(dfs, ignore_index = True)
    return df
    
def main():
    args = parse_arguments()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    CMS = {
        "font.family": "sans-serif",
        "mathtext.fontset": "custom",
        "mathtext.rm": "TeX Gyre Heros",
        "mathtext.bf": "TeX Gyre Heros:bold",
        "mathtext.sf": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic",
        "mathtext.tt": "TeX Gyre Heros",
        "mathtext.cal": "TeX Gyre Heros",
        "mathtext.default": "regular",
        "figure.figsize": (10.0, 10.0),
        "font.size": 26,
        "axes.labelsize": "medium",
        "axes.unicode_minus": False,
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.fontsize": "small",
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.5,
        "xtick.direction": "in",
        "xtick.major.size": 12,
        "xtick.minor.size": 6,
        "xtick.major.pad": 6,
        "xtick.top": True,
        "xtick.major.top": True,
        "xtick.major.bottom": True,
        "xtick.minor.top": True,
        "xtick.minor.bottom": True,
        "xtick.minor.visible": True,
        "ytick.direction": "in",
        "ytick.major.size": 12,
        "ytick.minor.size": 6.0,
        "ytick.right": True,
        "ytick.major.left": True,
        "ytick.major.right": True,
        "ytick.minor.left": True,
        "ytick.minor.right": True,
        "ytick.minor.visible": True,
        "grid.alpha": 0.8,
        "grid.linestyle": ":",
        "axes.linewidth": 2,
        "savefig.transparent": False,
    }
    plt.style.use(CMS)

    pred_dfs = {}
    for name, ckpt_cfg in cfg["models"].items():
        if ckpt_cfg["class"] == "MLP":
            model = MLPKD.load_from_checkpoint(ckpt_cfg["ckpt"]).cuda()
        elif ckpt_cfg["class"] == "DeepSet":
            model = DeepSetKD.load_from_checkpoint(ckpt_cfg["ckpt"]).cuda()
        else:
            raise NotImplementedError(f"model {name} is not implemented")
        pred_dfs[name] = get_predictions(model)


    fig, (ax) = plt.subplots(figsize = (8, 8), ncols = 1, nrows = 1)
    for name, df in pred_dfs.items():
        plot_roc(name, df, ax)
    ax.legend()
    ax.grid(True, which="both", ls="--", color='0.65')
    ax.set_yscale("log")
    ax.set_ylim([1, 3e4])
    ax.set_xlabel(r"Signal efficiency $\epsilon_s$")
    ax.set_ylabel(r"Background rejection $1/\epsilon_b$")
    fig.tight_layout()
    fig.savefig("plots/roc.pdf")
    plt.close(fig)

    fig, (ax) = plt.subplots(figsize = (12, 8), ncols = 1, nrows = 1)
    for name, df in pred_dfs.items():
        plot_boost(name, df, ax)
    ax.legend(fontsize = 'x-small')
    ax.grid(True, ls="--", color='0.65')
    ax.set_ylim([0.5, 1])
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig("plots/boost.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()