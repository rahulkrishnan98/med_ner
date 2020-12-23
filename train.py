import pandas as pd
from datetime import datetime
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import dataset 
import model
import os
import numpy as np
import joblib

def extract(data, sent_ind, tokens, tags):
    tokens = data.groupby(sent_ind)[tokens].apply(list).values
    tags = data.groupby(sent_ind)[tags].apply(list).values
    temp, max_sequence_len= [], 0

    for t in tags:
        if len(t)> max_sequence_len:
            max_sequence_len= len(t)
        temp.extend(t)
    class_count= len(set(temp))

    del temp
    return tokens, tags, class_count, max_sequence_len

if __name__ == "__main__":
    with open("config.json", "r") as fp:
        config = json.load(fp)
    strftime = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    working_dir = Path(config["working_dir"]) / strftime
    os.mkdir(working_dir)
    os.mkdir(os.path.join(working_dir,"model_ckpt"))
    train = pd.read_csv(
        config['training_data'],
        error_bad_lines= True
    )
    
    sentences, tags, class_count, max_sequence_len= extract(
        train,
        sent_ind="Sentence_Index",
        tokens = "Token",
        tags = "Tag"
    )

    #overwrite if present 
    if config["max_sequence_len"]:
        max_sequence_len= config["max_sequence_len"]

    dataSetObj = dataset.DataSet(
        sentences,
        tags,
        class_count,
        max_sequence_len
    )

    sentences, tags, all_words = dataSetObj.getData()
    meta = dataSetObj.getMeta()
    meta["max_sequence_len"] = max_sequence_len

    joblib.dump(
        meta, 
        os.path.join(working_dir,"meta.bin")
    )

    sent_train, sent_test, tag_train, tag_test = train_test_split(
        sentences, tags, test_size=0.2, random_state=42
    )

    model = model.recurrent_model(
        class_count,
        vocab_size= len(set(all_words)) + 1,
        embedding_dim= 100,
        max_sequence_len= max_sequence_len,
    )

    model_callbacks= [
        tf.keras.callbacks.TensorBoard(
            log_dir= os.path.join(working_dir, "logs"),
            write_graph=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath= os.path.join(working_dir,"model_ckpt",'{epoch:02d}.hdf5'), 
            monitor='val_loss',  
            save_best_only=False,
            save_weights_only=False, 
            mode='auto', 
            save_freq='epoch'
        )
    ]

    history = model.fit(
        sent_train, np.array(tag_train), 
        epochs = config["epochs"],
        batch_size = config["batch_size"],
        validation_data = (sent_test, np.array(tag_test)),
        callbacks = model_callbacks
    )
    