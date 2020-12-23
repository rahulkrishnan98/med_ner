import tensorflow as tf 
import pandas as pd 
import json 
import joblib 
import inference
import train
import os
from tqdm import tqdm
tqdm.pandas()

if __name__ == "__main__":
    with open("config.json", "r") as fp:
        config = json.load(fp)
    
    test_data = pd.read_csv(
        config['testing_data'],
        error_bad_lines= True
    )

    tokens, tags, _, _ = train.extract(
        test_data, 
        sent_ind="Sentence_Index",
        tokens = "Token",
        tags = "Tag"
    )

    tokens, tags = tokens[:1000], tags[:1000]
       
    meta = joblib.load(config['meta_path'])
    inferenceObj = inference.Inference(
        meta, 
        config["model_ckpt_path"]
    )
    
    result_df = pd.DataFrame(
        columns=["tokens","y_true","y_pred","confidence"]
    )

    result_df['tokens'] = tokens
    result_df['y_true'] = tags 

    result_df['y_pred'] = result_df['tokens'].progress_apply(
        lambda x: inferenceObj.predict(x)[:len(x)]
    )
    
    result_df.to_csv(
        os.path.join(
            config['prediction_save_path'], "predictions.csv"
        )
    )
