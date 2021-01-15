import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences

class Inference:
    def __init__(self, meta, model_path):
        '''
        input params
        model: saved model_path
        meta: word2idx and tag2idx mapping
        '''
        self.word2idx, self.tag2idx, self.max_sequence_len = meta["words"], meta["tags"], meta["max_sequence_len"]
        self.words = {index: key for key, index in self.word2idx.items()}
        self.tags = {index: key for key, index in self.tag2idx.items()}

        self.model =  tf.keras.models.load_model(
            model_path
        )
        print(self.model.summary())
    
    def predict(self,sentence, verbose=False):
        '''
        input param
        sentence: tokenized sentence
        '''

        #converting to required dims
        sentence = list(
            map(
                lambda x: self.word2idx.get(x, self.word2idx['UNK']), sentence
            )
        )

        sentence = pad_sequences(
            maxlen=self.max_sequence_len, 
            sequences= [sentence], 
            padding="post",
            value= self.word2idx['PAD']
        )

        prediction= self.model(
            sentence
        )
        prediction = np.argmax(prediction, axis=-1)
        # confidence = np.max(prediction, axis=-1)

        prediction = list(
            map(
                lambda x: self.tags[x], prediction[0]
            )
        )

        if verbose:
            for word, tag in zip(sentence[0], prediction):
                print(self.words[word], tag)
        return prediction

