# Implementation of the full weighted loss for Species Distribution Modeling

From the paper: [***On the selection and effectiveness of pseudo-absences for species distribution modeling with deep learning***](https://www.sciencedirect.com/science/article/pii/S1574954124001651?utm_campaign=STMJ_219742_AUTH_SERV_PA&utm_medium=email&utm_acid=289699847&SIS_ID=&dgcid=STMJ_219742_AUTH_SERV_PA&CMX_ID=&utm_in=DM476774&utm_source=AC_)

The implementation of the full weighted loss function $\mathcal{L}_{\text{full-weighted}}$, as well as other baseline loss functions, can be found in the `losses.py` file.


### Installation

```pip install -r requirements.txt```

### Training a model with the full weighted loss function

```python train_model.py --region SWI```

## How to cite
```
@article{zbinden2024selection,
    title = {On the selection and effectiveness of pseudo-absences for species distribution modeling with deep learning},
    author = {Robin Zbinden and Nina {van Tiel} and Benjamin Kellenberger and Lloyd Hughes and Devis Tuia},
    journal = {Ecological Informatics},
    volume = {81},
    pages = {102623},
    year = {2024},
    issn = {1574-9541}
}
```

We also evaluate the loss on other datasets (GeoLifeCLEF and iNaturalist) in this paper:
```
@article{zbinden2024imbalance,
  title={Imbalance-aware Presence-only Loss Function for Species Distribution Modeling},
  author={Zbinden, Robin and van Tiel, Nina and Ru{\ss}wurm, Marc and Tuia, Devis},
  journal={ICLR 2024 Workshop: Tackling Climate Change with Machine Learning},
  year={2024}
}
```
