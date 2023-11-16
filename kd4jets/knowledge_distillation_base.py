# system imports
import sys

# 3rd party imports
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import yaml
import matplotlib
import matplotlib.pyplot as plt
import wandb
from sklearn.decomposition import PCA

# from .dataset import retrieve_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
matplotlib.use('agg')


class KnowledgeDistillationBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module
        """
        self.save_hyperparameters(hparams)
        
        # Initialize Datasets
        self.loaders = retrieve_dataloaders(hparams["batch_size"])
        
        # Initialize models
        self.student = self.get_student(hparams)
        self.teacher = self.get_teacher(hparams)
        
        # Disable Gradients for Teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Initialize Prediction Outputs:
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.pca = PCA(n_components = 2)
        
        # Initialize counter
        self.register_buffer("num_iter", torch.tensor(0), persistent=True)
        
    def train_dataloader(self):
        return self.loaders["train"]

    def val_dataloader(self):
        return self.loaders["valid"]

    def test_dataloader(self):
        return self.loaders["test"]
        
    def get_student(self, hparams):
        """
        return a torch.nn.Module model that is the student model. The forward() function should return logits
        """        
        raise NotImplementedError("You need to implement the method!")
        
    def get_teacher(self, hparams):
        """
        return a torch.nn.Module model that is the teacher model. The forward() function should return logits
        """       
        raise NotImplementedError("You need to implement the method!")
        
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.student.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"]
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    @property
    def decay(self):
        if "auxiliary_teacher" in self.hparams:
            return 1 - torch.clamp(self.num_iter/self.hparams["auxiliary_teacher"], 0, 1)
        else:
            return 1
    
    def mse_loss(self, label, from_student, from_teacher):
        return F.mse_loss(from_student, from_teacher)
    
    def dist_loss(self, label, from_student, from_teacher):
        
        student_dist_matrix = torch.sqrt(1e-6 + (from_student.unsqueeze(0) - from_student.unsqueeze(1)).square().sum(-1))
        teacher_dist_matrix = torch.sqrt(1e-6 + (from_teacher.unsqueeze(0) - from_teacher.unsqueeze(1)).square().sum(-1))
        
        return F.mse_loss(student_dist_matrix, teacher_dist_matrix)
    
    def training_step(self, batch, batch_idx):
        self.num_iter += 1
        labels = batch["is_signal"].float()
        
        if self.hparams.get("boost", False):
            batch = boost_batch(batch, self.hparams["boost"] * torch.rand_like(labels))
        
        if self.hparams["T"] > 0:
            with torch.no_grad():
                teacher_logits, teacher_features = self.teacher(batch)
        logits, student_features = self.student(batch)
        
        hard_loss = F.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0], labels)
        self.log("training_hard_loss", hard_loss)

        if self.hparams["T"] > 0:
            soft_loss = F.kl_div(
                F.log_softmax(logits/self.hparams["T"], dim = -1),
                F.softmax(teacher_logits/self.hparams["T"], dim = -1),
                reduction='batchmean'
            )
            weight = self.hparams["lambda"] * self.decay
            loss = (1-weight) * hard_loss + weight * (self.hparams["T"] ** 2) * soft_loss
            self.log("training_soft_loss", soft_loss)
        else:
            loss = hard_loss
        
        
        for name, (feature_name, weight) in self.hparams["guidance"].items():
            guidance_loss = getattr(self, f"{name}_loss")(labels, student_features[feature_name], teacher_features[feature_name])
            loss += self.decay * weight * guidance_loss
            self.log(f"training_{name}_{feature_name}_guidance", guidance_loss)
            
        self.log("training_loss", loss)
        
        return loss
    
    def shared_evaluation(self, batch, batch_idx, log=False):
        
        labels = batch["is_signal"].float()
        
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(batch)
        logits, student_features = self.student(batch)
        logits[logits != logits] = 0
        
        hard_loss = F.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0], labels)
        self.log("val_hard_loss", hard_loss)

        if self.hparams["T"] > 0:
            soft_loss = F.kl_div(
                F.log_softmax(logits/self.hparams["T"], dim = -1),
                F.softmax(teacher_logits/self.hparams["T"], dim = -1),
                reduction='batchmean'
            )
            weight = self.hparams["lambda"] * self.decay
            loss = (1-weight) * hard_loss + weight * (self.hparams["T"] ** 2) * soft_loss
            self.log("val_soft_loss", soft_loss)
        else:
            loss = hard_loss
            
        feature_names = ["rep", "emb"]
        for name, (feature_name, weight) in self.hparams["guidance"].items():
            guidance_loss = getattr(self, f"{name}_loss")(labels, student_features[feature_name], teacher_features[feature_name])
            loss += self.decay * weight * guidance_loss
            self.log(f"val_{name}_{feature_name}_guidance", guidance_loss)
            
        self.log("val_loss", loss)
        
        # store predictions
        prob = F.softmax(logits, dim = -1)
        
        # Boosting Test
        beta = torch.as_tensor(np.linspace(0, 1, self.hparams["batch_size"]//4, endpoint = False).repeat(4))
        boosted_logits, _ = self.student(boost_batch(batch, beta))
        boosted_pred = boosted_logits.argmax(-1)
        
        return (
            labels.cpu().numpy(),
            prob.cpu().numpy(),
            student_features["rep"].cpu().numpy(),
            boosted_pred.cpu().numpy(),
            beta.numpy()
        )

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self.shared_evaluation(batch, batch_idx, log=True))

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_evaluation(batch, batch_idx, log=True))
    
    def make_pca_plot(self, labels, student_emb):
        fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi = 100, facecolor='w', edgecolor='k')

        colors = ['#a1c9f4' if l == 1 else '#8de5a1' for l in labels] 
            
        self.pca.fit(student_emb)
        student_PCA = self.pca.transform(student_emb).T

        ax.scatter(student_PCA[0], student_PCA[1], s = 1, color = colors)
        ax.set_title("student's PCA")
        
        fig.tight_layout()
        img = wandb.Image(fig, caption="Dimension Reduction of penultimate Layer's output")
        fig.clear()
        plt.close()
        return img    
    
    def make_wandb_plot(self, x_data, y_data, xtitle, ytitle, title):
        table = wandb.Table(data=[[x, y] for (x, y) in zip(x_data, y_data)], columns = [xtitle, ytitle])
        return wandb.plot.line(table, xtitle, ytitle, title=title)
    
    def shared_on_epoch_end_eval(self, preds):
        # Transpose list of lists
        labels, prob, student_emb, boosted_pred, beta = map(lambda x: np.concatenate(list(x), axis = 0), zip(*preds))
        df = pd.DataFrame({"beta": beta, "acc": (boosted_pred == labels)})
        acc_beta = df.groupby("beta").mean()
        # Evaluate model performance
        auc = roc_auc_score(labels, prob[:, 1])
        acc = np.mean(labels == prob.argmax(-1))
        fpr, tpr, _ = roc_curve(labels, prob[:, 1])
        rej = 1./np.interp([0.3, 0.5], tpr, fpr)
        self.log_dict({
            "auc": auc,
            "acc": acc,
            "rej_30": rej[0], 
            "rej_50": rej[1],           
            "invariance score": (2 * acc_beta.mean().item() - 1)/(2 * acc_beta.max().item() - 1),
        })
        
        wandb.log({
            "Penultimate layer outputs": self.make_pca_plot(labels[:10000], student_emb[:10000]),
            "ROC curve": self.make_wandb_plot(
                np.interp(np.linspace(0, 1, 1001), tpr, fpr),
                np.linspace(0, 1, 1001),
                "False Positive Rate",
                "True Positive Rate",
                "ROC curve"
            ),
            "Accuracy vs. beta": self.make_wandb_plot(acc_beta.index, acc_beta["acc"], "beta", "Accuracy", "Accuracy vs. beta")
        })

        
    def on_validation_epoch_end(self):
        self.shared_on_epoch_end_eval(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        self.shared_on_epoch_end_eval(self.test_step_outputs)
        self.test_step_outputs.clear()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        
def boost_jets(pmu, mask, beta, mode = "x"):
    """
    pmu: four momenta (E, px, py, pz) in shape of [N, P, 4]
    mask: particle mask in shape of [N, P], bool tensor
    beta: relative speed of the new frame. float tensor of shape [N]
    mode: can be either x, z, or jet; the axis to boost along
    """
    gamma = torch.rsqrt(1 - beta.square()).view(-1, 1, 1)
    beta = beta.view(-1, 1, 1)
    if mode == "x":
        lambda_pmu = torch.cat([
            gamma * pmu[:,:,[0]] - beta * gamma * pmu[:,:,[1]],
            - beta * gamma * pmu[:,:,[0]] + gamma * pmu[:,:,[1]],
            pmu[:,:,2:4]
        ], dim = -1)
    if mode == "z":
        lambda_pmu = torch.cat([
            gamma * pmu[:,:,[0]] - beta * gamma * pmu[:,:,[3]],
            pmu[:,:,1:3]
            - beta * gamma * pmu[:,:,[3]] + gamma * pmu[:,:,[0]],
        ], dim = -1)
    if mode == "jet":
        p_jets = (pmu * mask.float()[:, :, None]).sum(1) # [N, 4]
        beta_vectors = beta * F.normalize(p_jets[:, 1:4], dim = -1).view(-1, 3, 1) # [N, 3, 1]
        bx, by, bz = beta_vectors[:,[0]], beta_vectors[:,[1]], beta_vectors[:,[2]] # [N, 1, 1]
        lambda_pmu = torch.cat([
              gamma * pmu[:,:,[0]] - gamma * bx * pmu[:,:,[1]] - gamma * by * pmu[:,:,[2]] - gamma * bz * pmu[:,:,[3]],
            - gamma * bx * pmu[:,:,[0]] + (1 + (gamma - 1) * bx * bx / beta.square()) * pmu[:,:,[1]] + (gamma - 1) * by * bx / beta.square() * pmu[:,:,[2]] + (gamma - 1) * bz * bx / beta.square() * pmu[:,:,[3]],
            - gamma * by * pmu[:,:,[0]] + (gamma - 1) * by * bx / beta.square() * pmu[:,:,[1]] + (1 + (gamma - 1) * by * by / beta.square()) * pmu[:,:,[2]] + (gamma - 1) * bz * by / beta.square() * pmu[:,:,[3]],
            - gamma * bz * pmu[:,:,[0]] + (gamma - 1) * bz * bx / beta.square() * pmu[:,:,[1]] + (gamma - 1) * by * bz / beta.square() * pmu[:,:,[2]] + (1 + (gamma - 1) * bz * bz / beta.square()) * pmu[:,:,[3]],
        ], dim = -1)
    lambda_pmu[beta.view(-1) == 0] = pmu[beta.view(-1) == 0]
    return lambda_pmu.masked_fill_((~mask[:, :, None]) | (lambda_pmu != lambda_pmu), 0)

def boost_batch(batch, beta):
    lambda_pmu = boost_jets(batch["Pmu"], batch["atom_mask"], beta.to(batch["Pmu"].device), mode = "x")
    new_batch = batch.copy()
    new_batch["Pmu"] = torch.cat([new_batch["Pmu"][:, :2], lambda_pmu[:, 2:]], dim = 1)
    return new_batch