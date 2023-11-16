# Third party import
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
import torch
import yaml

# Local import
from jetrep.models import CLIP

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--cfg", type = str, required = True)
    parser.add_argument("--accelerator", type = str, default="gpu")
    parser.add_argument("--devices", type = int, default=1)
    parser.add_argument("--epochs", type = int, default=100)
    parser.add_argument("--train_batches", type = int, default=10000)
    parser.add_argument("--val_batches", type = int, default=2000)
    parser.add_argument("--gradient_clip_val", type = float, default=0.5)
    parser.add_argument("--num_sanity_val_steps", type = int, default=0)
    parser.add_argument("--log_period", type = int, default=50)
    parser.add_argument("--log_path", type = str, default=None)
    parser.add_argument("--project_name", type = str, default="representation learning")
    parser.add_argument("--anomaly_detection_cfg", type = str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.anomaly_detection_cfg:
        with open(args.anomaly_detection_cfg, 'r') as f:
            cfg["anomaly_detection_cfg"] = yaml.safe_load(f)
            cfg["anomaly_detection_cfg"].update({
                "d_emb": cfg["d_emb"],
                "train_index": cfg["train_index"]
            })
    if cfg["model"] == "CLIP":
        model = CLIP(
            gradient_clip_val = args.gradient_clip_val,
            **cfg
        )
    else:
        raise NotImplementedError("model specified is not implemented")
        
    torch.compile(model)
    torch.set_float32_matmul_precision('medium')
    logger = WandbLogger(
        project = args.project_name, 
        save_dir = args.log_path
    )
    
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        limit_train_batches = args.train_batches,
        limit_val_batches = args.val_batches,
        num_sanity_val_steps = args.num_sanity_val_steps,
        logger=logger,
        callbacks=[ModelCheckpoint()],
        log_every_n_steps = args.log_period,
        default_root_dir = args.log_path
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()