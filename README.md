# [TEPCAM: Prediction of T-cell receptor–epitope binding specificity via interpretable deep learning](https://doi.org/10.1002/pro.4841)


Implementation of TEPCAM, a binary classification model for TCR-beta-CDR3 and epitope.     
This repository contains processed data, code and checkpoint.

![image](pics/model.png)

## Requirements
TEPCAM is constructed using python 3.8.16. The detail dependencies are recorded in `requirements.txt`.    

To install from the [requirements.txt](requirements.txt), using:     

```bash
pip install -r requirements.txt
```   

Using a conda virtual environment is highly recommended.

``` console
conda create -f env.yaml
```

## Model Training
For training TEPCAM, run command below:
```commandline
python ./scripts/train.py \
        --input_file="./Data/TEP-merge.csv" \
        --model_name="tepcam_test" \
        --epoch=30 \
        --learning_rate=5e-4 \
        --GPU_num=0
```
## Model Evaluation
Model evalutation using [test.py](./scripts/test.py), run command below:
```commandline
python ./scripts/test.py 
        --file_path="./Data/ImmuneCODE.csv" 
        --model_path="./ckpts/tepcam_test.pt" 
        --output_file="./result/output.csv" 
        --metric_file="./result/metrics.txt"
```

## Retraining TEPCAM

If you want to use your own data to retrain TEPCAM, the data(.csv) file must contain these columns:

* `TCR`: TCR sequence (e.g., CASSQSPGFTGNEQFF)
* `epitope`: Epitope sequence (e.g., GILGFVFTL)
* `Label`: Binary binding label (1=binding, 0=non-binding)


## Citation
Chen J, Zhao B, Lin S, Sun H, Mao X, Wang M, et al. TEPCAM: Prediction of T-cell receptor–epitope binding specificity via interpretable deep learning. Protein Science. 2024; 33(1):e4841. https://doi.org/10.1002/pro.4841
