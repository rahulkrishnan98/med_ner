'''
model.py -  BiLSTM model
vocab.py - Builds vocabulary
driver.py - Runs training and testing loop
'''
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from vocab import BuildVocab
import utils
import random
import json
import os
import model as Model

def train(args, device, writer):
    """ Training BiLSTMCRF model
    Args:
        args: config file
    """
    sent_vocab = BuildVocab.load(
        os.path.join(args['SENT_VOCAB_PATH'], "sent_vocab.json")
    )
    tag_vocab = BuildVocab.load(
        os.path.join(args['TAG_VOCAB_PATH'], "tag_vocab.json")
    )
    train_data, dev_data = utils.generate_train_dev_dataset(
        args['train_file'], sent_vocab, tag_vocab,
        train_proportion= 0.8,
        sent_ind= config['sentence_id_col'],
        tokens= config['tokens_col'],
        tags= config['tags_col']
    )
    print('num of training examples: %d' % (len(train_data)))
    print('num of development examples: %d' % (len(dev_data)))

    max_epoch = int(args['max_epoch'])
    log_every = int(args['log_freq'])
    validation_every = int(args['validation_freq'])
    model_save_path = os.path.join(args['model_save_path'], "model.pth")
    optimizer_save_path = os.path.join(args['optimizer_save_path'], "optimizer.pth")

    min_dev_loss = float('inf')
    patience, decay_num = 0, 0

    model = Model.BiLSTMCRF(
        sent_vocab, tag_vocab, 
        float(args['dropout_rate']), 
        int(args['embed_size']),
        int(args['hidden_size'])
    ).to(device)

    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['lr']))
    train_iter = 0  # train iter num
    record_loss_sum, record_tgt_word_sum, record_batch_size = 0, 0, 0  # sum in one training log
    cum_loss_sum, cum_tgt_word_sum, cum_batch_size = 0, 0, 0  # sum in one validation log
    record_start, cum_start = time.time(), time.time()

    #training loop
    print('start training...')
    for epoch in range(max_epoch):
        for sentences, tags in utils.batch_iter(train_data, batch_size=int(args['batch_size'])):
            train_iter += 1
            current_batch_size = len(sentences)
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[tag_vocab.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args['clip_max_norm']))
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_tgt_word_sum += sum(sent_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_tgt_word_sum += sum(sent_lengths)

            if train_iter % log_every == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_tgt_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_tgt_word_sum = 0, 0, 0
                record_start = time.time()

            if train_iter % validation_every == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, cum_tgt_word_sum / (time.time() - cum_start),
                       cum_loss_sum / cum_batch_size, time.time() - cum_start))
                cum_loss_sum, cum_batch_size, cum_tgt_word_sum = 0, 0, 0

                dev_loss = cal_dev_loss(model, dev_data, args['batch_size'], sent_vocab, tag_vocab, device)
                if dev_loss < min_dev_loss * float(args['patience_threshold']):
                    min_dev_loss = dev_loss
                    #save only if there is improvement in val_loss
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                    patience = 0
                else:
                    patience += 1
                    if patience == int(args['max_patience']):
                        decay_num += 1
                        if decay_num == int(args['max_decay']):
                            print('Early stop. Save result model to %s' % model_save_path)
                            return
                        lr = optimizer.param_groups[0]['lr'] * float(args['lr_decay'])
                        #TODO: check for faster tweaks here
                        model = Model.BiLSTMCRF.load(model_save_path, device)
                        optimizer.load_state_dict(torch.load(optimizer_save_path))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, dev_loss, patience, decay_num))
                cum_start = time.time()
                if train_iter % log_every == 0:
                    record_start = time.time()
        #log metrics            
        writer.add_scalar(
            'Loss/Train', record_loss_sum, epoch
        )
        writer.add_scalar(
            'Loss/Validation', cum_loss_sum, epoch
        )
    print('Reached %d epochs, Save result model to %s' % (max_epoch, model_save_path))

def test(args, device):
    """ Testing the model
    Args:
        args: config json
    """
    sent_vocab = BuildVocab.load(os.path.join(args['SENT_VOCAB_PATH'], "sent_vocab.json"))
    tag_vocab = BuildVocab.load(os.path.join(args['TAG_VOCAB_PATH'], "tag_vocab.json"))
    sentences, tags = utils.getData(
        args['test_file'],
        sent_ind= args['sentence_id_col'],
        tokens= args['tokens_col'],
        tags= args['tags_col']
    )
    sentences = utils.words2indices(sentences, sent_vocab)
    tags = utils.words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags))
    print('num of test samples: %d' % (len(test_data)))

    model = Model.BiLSTMCRF.load(args['MODEL_CKPT'], device)
    print('start testing...')
    print('using device', device)

    result_file = open(args['prediction_save_file'], 'w')
    model.eval()
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(test_data, batch_size=int(args['batch_size']), shuffle=False):
            padded_sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            predicted_tags = model.predict(padded_sentences, sent_lengths)
            for sent, true_tags, pred_tags in zip(sentences, tags, predicted_tags):
                sent, true_tags, pred_tags = sent[1: -1], true_tags[1: -1], pred_tags[1: -1]
                for token, true_tag, pred_tag in zip(sent, true_tags, pred_tags):
                    #make sure to write after every prediction (since it is time consuming)
                    result_file.write(' '.join([sent_vocab.id2word(token), tag_vocab.id2word(true_tag),
                                                tag_vocab.id2word(pred_tag)]) + '\n')
                result_file.write('\n')

def cal_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    """ 
    Returns:
        the average loss on the dev data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        #we need batching here again, since the batch we did earlier was exclusive to train loop
        for sentences, tags in utils.batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[sent_vocab.PAD], device)
            batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences


if __name__ == "__main__":
    with open("config.json", "r") as fp:
        config = json.load(fp)
    
    #step_0: Create all sub_dir
    dir_list = [
        config["SENT_VOCAB_PATH"], config["model_save_path"], config['optimizer_save_path']
    ]
    utils.createDir(dir_list)

    #step_1: Choose device- cpu or cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #step_2: Setup tf_board logs
    writer = SummaryWriter(log_dir= config['log_dir'])

    #train / test
    if config["RUN_MODE"]== "test":
        test(config, device)
    else:
        train(config, device, writer)