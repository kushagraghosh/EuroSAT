## Goals
Trained a ResNet50 model on the EuroSAT satellite imagery dataset w/ PyTorch. Analyzed the model's encoder by visualizing linear interpolations within the embedding space to illustrate the semantic separation in the learned feature representations.

## Dataset
This project utilizes the EuroSAT dataset, which offers satellite images categorized into 10 distinct classes. The dataset is available for public access and can be found on the [EuroSAT GitHub repository](https://github.com/phelber/EuroSAT). I split the data into a training and testing set with an 80%-20% ratio.

The dataset is organized in the following directory hierarchy:

```
Train_Test_Splits
│
├── train
│   ├── AnnualCrop
│   ├── [Other Classes]
│
└── test
    ├── AnnualCrop
    ├── [Other Classes]
```

Make sure that the data path in the config.yaml is correct.

use pip to install all the requirements.
```
pip install -r requirements.txt
```

## Linear Interpolation In Latent Space
Showing nearest neighbors at each interpolation step from left (Industrial) to right (Forest). Inspired by plot in page 7 of [Tile2Vec](https://doi.org/10.48550/arXiv.1805.02855) paper.

![Matplotlib plot](https://github.com/kushagraghosh/EuroSAT/blob/master/reports/figures/linear_interpolation_of_embeddings.png?raw=true)
