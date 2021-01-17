# MED_NER - For chemical/ protein extraction

The repository is divided into two, *pytorch/* contains updated code and the following README.md will explain steps to reproduce code + run.
[The keras code is outdated and no readme is maintained for the same]


# Running pytorch NER code
## Setting up environment
 To set up the environment, simply install ***pytorch/requirements.txt*** via pip in your virtual env. 
 

    pip install -r requirements.txt

 
## Step 1 : Create Vocabulary

Move inside *pytorch* folder and run vocab.py, all necessary changes need to be made to config.json [no changes need to be made to any of the source files]

    python vocab.py

## Step 2 : Train a model 

Training a model is via driver.py file and it also saves the best model. To change this behavior make necessary changes to the *train()* method. 

    python driver.py

Setting mode to ***train*** in *config.json* [Default] runs train loop from driver.py. To run model inference or test, change this json param to ***test***.

