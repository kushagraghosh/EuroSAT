## Dataset
This project utilizes the EuroSAT dataset, which offers satellite images categorized into 10 distinct classes. The dataset is available for public access and can be found on the [EuroSAT GitHub repository](https://github.com/phelber/EuroSAT). For our specific project, the data is split into a training and testing set with an 80%-20% ratio.

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

you need to download it and unzip the file. Then adjust your data path in the config.yaml

## Requirements
matplotlib
numpy
pandas
scikit-learn
seaborn
torch
torchvision

use pip to install all the requirements.
```
pip install -r requirements.txt