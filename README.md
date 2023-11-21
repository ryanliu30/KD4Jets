<div align="center">

# Efficient and Robust Jet Tagging at the LHC with Knowledge Distillation

[Machine Learning and the Physical Sciences Workshop, NeurIPS 2023 Presentation](https://nips.cc/virtual/2023/76169)
    
[arXiv paper]()

[Author Contact](mailto:liuryan30@berkeley.edu)

</div>

Welcome to code repository for Efficient and Robust Jet Tagging at the LHC with Knowledge Distillation.

## Installation 
```
git clone https://github.com/ryanliu30/KD4Jets.git --recurse-submodules
cd KD4Jets
conda create -n KD4Jets python=3.11
conda activate KD4Jets
pip install -r requirements.txt
pip install -e .
```
Then, please follow the instructions in [LorentzNet](https://github.com/sdogsq/LorentzNet-release.git) to get the training data from [OSF](https://osf.io/7u3fk/?view_only=8c42f1b112ab4a43bcf208012f9db2df) and put them under the `data` directory.
## Usage
To begin with, run the following command:
```
python train.py --cfg experiments/deepset.yaml
```
This will train a deepset jet tagging model from scratch. To train with knowledge distillation, run:
```
python train.py --cfg experiments/deepsetKD.yaml
```
Note that the only difference of these two config file is that $T$ was set to $-1$ for the one trained from scratch and the knowledge distillation one has $T>0$. There are two other configs `MLP.yaml` and `MLPKD.yaml` that can be run using the same command.

To evaluate the models, run
```
python eval.py --cfg plots/example.yaml
```
Where in the `yaml` file you should specify the checkpoint to use. They should locate under the `LorentzNetKD` folder. We also supplied four model checkpoints from our training that can be used out of the box.
## Citation
If you use this work in your research, please cite:
```
BibTex here
```
