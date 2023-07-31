# TEPCAM
Prediction of T cell receptor-epitope binding specificity via interpretable deep learning
![image](pics/model.png)

## Requirements
TEPCAM is constructed using python 3.8.16. The detail dependencies are recorded in `requirements.txt`.    

To install from the [requirements.txt](requirements.txt), using:     

```bash
pip install -r requirements.txt
```   

Using a conda virtual environment is highly recommended.

``` console
conda create -n TEPCAM
conda activate TEPCAM
conda install -r requirements.txt
```

## Model Training
For training TEPCAM, run command below:
```commandline
python ./scripts/train.py --input_file="./Data/TEP-merge.csv" --model_name="TEPCAM" --epoch=30 --learning_rate=5e-4 --GPU_num=0
```
## Model Evaluation
Model evalutation using [test.py](./scripts/test.py), run command below:
```commandline
python ./scripts/test.py 
--file_path="./Data/ImmuneCODE.csv" 
--model_path="./model/TEPCAM_test.pt" 
--output_file="./output.csv" 
--metric_file="./metric_file.csv"
```
