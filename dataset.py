from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DataSet:
    def __init__(
        self,
        sentences,
        tags,
        class_count,
        max_sequence_len
    ):
        #sentences ['this','is','an','example',..]
        #tags ['tag1','tag2'..]
        self.max_sequence_len = max_sequence_len
        self.sentences = sentences
        self.tags = tags
        self.class_count= class_count
        self.all_words, self.all_tags = [], []

        for _, sent in enumerate(self.sentences):
            self.all_words.extend(sent)
        
        self.all_words.append("UNK")
        self.all_words.append("PAD")
        self.all_words= set(self.all_words)
        

        for _, tag in enumerate(self.tags):
            self.all_tags.extend(tag)
        self.all_tags = set(self.all_tags)
        
        self.word2idx = {
            word: index for index, word in enumerate(self.all_words)
        }
        self.tag2idx = {
            tag: index for index, tag in enumerate(self.all_tags)
        }
        
        self.tag2idx['O']= self.class_count 

        #padding
        self.padding()

    def __len__(self):
        return len(self.sentences)

    def padding(self):
        #word2idx + pad
        #tag2idx + pad
        for ind, (sent, tag) in enumerate(zip(self.sentences, self.tags)):
            self.sentences[ind] = list(
                map(lambda x: self.word2idx[x], sent)
            )
            self.tags[ind] = list(
                map(lambda x: self.tag2idx[x], tag)
            )
        self.sentences= pad_sequences(
            maxlen=self.max_sequence_len, 
            sequences=self.sentences, 
            padding="post",
            value= self.word2idx['PAD']
        )
        self.tags= pad_sequences(
            maxlen=self.max_sequence_len, 
            sequences=self.tags, 
            padding="post", 
            value= self.tag2idx["O"]
        )
        self.tags = [to_categorical(sent) for sent in self.tags]
    
    def getData(self):
        return self.sentences, self.tags, self.all_words
    def getMeta(self):
        return {"words": self.word2idx, "tags": self.tag2idx}