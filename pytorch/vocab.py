from itertools import chain
from collections import Counter
import json
import os
import utils


class BuildVocab:
    '''
    input_params
    words- List[List]
    tags - List[List]

    output
    returns vocab object
    that has mapping for every word - idx pair
    __id2word is simply an array (each index is the id for the word)
    use __word2id instead
    '''
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word
    
    #helper methods
    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]
    
    @staticmethod
    def build(
        data,
        max_vocab_size, 
        frequency_if_exceeds
    ):
        '''
        data - list of list of strings
        max_vocab_size - integer 
        frequency_if_exceeds - integer
        _______________________________________
        if number of unique words exceeds given 
        max_vocab_size then take only number of
        words = frequency_if_exceeds
        '''

        word_counts = Counter(chain(*data)) #chain gets one iterable
        if len(word_counts)<=max_vocab_size:
            valid_words = [w for w, d in word_counts.items()]
        else: 
            valid_words = [w for w, d in word_counts.items() if d >= frequency_if_exceeds]
        valid_words = sorted(
            valid_words, key=lambda x: word_counts[x], reverse=True #highest freq first
        )
        valid_words = valid_words[: max_vocab_size]
        #special Symbols
        valid_words += ['<PAD>']
        valid_words += ['<UNK>']
        word2id = {w: idx for idx, w in enumerate(valid_words)}
        
        return BuildVocab(
            word2id= word2id,
            id2word= valid_words
        )

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump({'word2id': self.__word2id, 'id2word': self.__id2word}, f, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf8') as f:
            entry = json.load(f)
        return BuildVocab(word2id=entry['word2id'], id2word=entry['id2word'])

if __name__ == "__main__":
    with open("config.json", "r") as fp:
        config = json.load(fp)
    sentences, tags = utils.getData(
        file_path= config['train_file'],
        sent_ind= config['sentence_id_col'],
        tokens= config['tokens_col'],
        tags= config['tags_col']
    )
    sent_vocab = BuildVocab.build(sentences, int(config['max_size']), int(config['freq_cutoff']))
    tag_vocab = BuildVocab.build(tags, int(config['max_size']), int(config['freq_cutoff']))
    sent_vocab.save(
        os.path.join(
            config['SENT_VOCAB_PATH'],
            "sent_vocab.json"
        )
        
    )
    tag_vocab.save(
        os.path.join(
            config['TAG_VOCAB_PATH'],
            "tag_vocab.json"
        )
    )
