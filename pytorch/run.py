
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report

import argparse
import datetime
import math
import json
import os
import csv
import numpy as np
import logging
import random

from char_lstm_tagger import CharLSTMTagger
from lstm_tagger import LSTMTagger
from char_cnn_tagger import CharCNNTagger
from mtl_wrapper import MTLWrapper
try:
    import load_data
    from utils import split_train_val
    from data_classes import write_sentences_to_excel
except:
    pass

UNKNOWN = 'UNKNOWN'
FLAT = "flat"
MTL = "multitask"
HIERARCHICAL = "hierarchical"
THRESHOLD = 2


def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return index


def get_max_prob_result(input, ix_to_tag):
    return ix_to_tag[get_index_of_max(input)]


def prepare_char_sequence(word, to_ix):
    idxs = []
    for i in range(len(word) - 1):
        curr_char = word[i]
        next_char = word[i+1]
        if curr_char == "'":  # treat letters followed by ' as single character
            continue
        if next_char == "'":
            char = curr_char+next_char
        else:
            char = curr_char
        idxs.append(get_index(char, to_ix))
    if word[-1] != "'":
        idxs.append(get_index(word[-1], to_ix))
    return idxs


def prepare_sequence_for_chars(seq, to_ix, char_to_ix, poses=None, pos_to_ix=None):
    res = []
    if pos_to_ix and poses:
        for (w, _), pos in zip(seq, poses):
            res.append((get_index(w, to_ix),
                        prepare_char_sequence(w, char_to_ix),
                        get_index(pos, pos_to_ix)))
    else:
        for w, _ in seq:
            res.append((get_index(w, to_ix), prepare_char_sequence(w, char_to_ix)))
    return res


def prepare_sequence_for_bpes(seq, to_ix, bpe_to_ix, poses=None, pos_to_ix=None):
    res = []
    if pos_to_ix and poses:
        for (w, bpes), pos in zip(seq, poses):
            res.append((get_index(w, to_ix), [get_index(bpe, bpe_to_ix) for bpe in bpes],
                        get_index(pos, pos_to_ix)))
    else:
        for w, bpes in seq:
            res.append((get_index(w, to_ix),
                        [get_index(bpe, bpe_to_ix) for bpe in bpes]))
    return res


def prepare_sequence_for_words(seq, to_ix, poses=None, pos_to_ix=None):
    idxs = []
    if pos_to_ix and poses:
        for (w, _), pos in zip(seq, poses):
            idxs.append((get_index(w, to_ix),
                        get_index(pos, pos_to_ix)))
    else:
        for w, _ in seq:
            ix = get_index(w, to_ix)
            idxs.append((ix,))
    return idxs


def prepare_target(seq, to_ix, field_idx):
    idxs = []
    for w in seq:
        ix = get_index(w[field_idx], to_ix)
        idxs.append(ix)
    return torch.LongTensor(idxs)


def reverse_dict(to_ix):
    # receives a dictionary with words/tags mapped to indices
    # and returns a dictionary mapping indices to words/tags
    ix_to = {}
    for k, v in to_ix.items():
        ix_to[v] = k
    return ix_to


def get_index(w, to_ix):
    return to_ix.get(w, to_ix[UNKNOWN])


def train(training_data, val_data, model_path, word_dict_path, char_dict_path,
          bpe_dict_path, tag_dict_path, frequencies, word_emb_dim, char_emb_dim,
          hidden_dim, dropout, num_kernels=1000, kernel_width=6, by_char=False,
          by_bpe=False, with_smoothing=False, cnn=False, directions=1, device='cpu',
          save_all_models=False, save_best_model=True, epochs=300, lr=0.1, batch_size=8,
          morph=None, weight_decay=0, loss_weights=(1,1,1,1,1), seed=42):

    # training data of shape: [(sent, tags), (sent, tags)]
    # where sent is of shape: [(word, bpe), (word, bpe)], len(sent) == number of words
    # and tags is of shape: [(pos, an1, an2, an3, enc),...], len(tags) == len(sent) == number of words

    field_names = ["pos", "an1", "an2", "an3", "enc"]

    model_path_parts = model_path.split(".")
    dict_path_parts = tag_dict_path.split(".")

    pos_training_data = [(sent, [tag_set[0] for tag_set in tags]) for sent, tags in training_data]
    logger.info(f"Number of sentences in training data: {len(pos_training_data)}")

    pos_model_path = model_path_parts[0] + "-pos." + model_path_parts[1]
    pos_dict_path = dict_path_parts[0] + "-pos." + dict_path_parts[1]

    word_to_ix, char_to_ix, bpe_to_ix, pos_to_ix = prepare_dictionaries(pos_training_data, with_smoothing, frequencies)
    torch.save(pos_to_ix, pos_dict_path)
    torch.save(word_to_ix, word_dict_path)
    torch.save(char_to_ix, char_dict_path)
    torch.save(bpe_to_ix, bpe_dict_path)

    val_sents = None

    if by_char:
        if val_data:
            val_sents = [prepare_sequence_for_chars(val_sent[0], word_to_ix,
                                                    char_to_ix)
                         for i, val_sent in enumerate(val_data)]
        train_sents = [prepare_sequence_for_chars(training_sent[0], word_to_ix,
                                                  char_to_ix)
                       for i, training_sent in enumerate(training_data)]
    elif by_bpe:
        if val_data:
            val_sents = [prepare_sequence_for_bpes(val_sent[0], word_to_ix, bpe_to_ix)
                         for i, val_sent in enumerate(val_data)]
        train_sents = [prepare_sequence_for_bpes(training_sent[0], word_to_ix,
                                                 bpe_to_ix)
                       for i, training_sent in enumerate(training_data)]
    else:
        if val_data:
            val_sents = [prepare_sequence_for_words(val_sent[0], word_to_ix)
                         for i, val_sent in enumerate(val_data)]
        train_sents = [prepare_sequence_for_words(training_sent[0], word_to_ix)
                       for i, training_sent in enumerate(training_data)]
    if val_data:
        logger.info(f"Number of sentences in val data: {len(val_sents)}")
    
        val_poses = [[prepare_target(tag_sets, pos_to_ix, field_idx=0).to(device=device)]
                     # inside a list for MTL wrapper purposes
                     for (val_sent, tag_sets) in val_data]
    else:
        val_poses = None
    train_poses = [[prepare_target(tag_sets, pos_to_ix, field_idx=0).to(device=device)]
                   # inside a list for MTL wrapper purposes
                   for (train_sent, tag_sets) in training_data]

    logger.info("Finished preparing POS data:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    if morph == MTL:

        model_path = model_path

        all_train_field_tags, all_val_field_tags, all_field_dicts = prepare_data_for_mtl(field_names, training_data,
                                                                                         val_data, device,
                                                                                         dict_path_parts)

        logger.info(f"Finished preparing MTL data:")
        logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

        mtl_model = train_tag(train_sents, val_sents, all_train_field_tags, all_val_field_tags, model_path,
                              word_to_ix, char_to_ix, bpe_to_ix, all_field_dicts, word_emb_dim, char_emb_dim,
                              hidden_dim, dropout, num_kernels, kernel_width, by_char, by_bpe, cnn, directions,
                              device, save_all_models, save_best_model, epochs, lr, batch_size,
                              weight_decay=weight_decay, loss_weights=loss_weights, seed=seed)
        return mtl_model

    logger.info("Preparing POS training data:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    pos_model = train_tag(train_sents, val_sents, train_poses, val_poses, pos_model_path, word_to_ix,
                          char_to_ix, bpe_to_ix, [pos_to_ix],
                          word_emb_dim, char_emb_dim, hidden_dim, dropout, num_kernels,
                          kernel_width, by_char, by_bpe, cnn, directions,
                          device, save_all_models, save_best_model, epochs, lr,
                          batch_size, weight_decay=weight_decay, seed=seed)

    if morph == FLAT or morph == HIERARCHICAL:

        pos_dict_size = 0

        if morph == HIERARCHICAL:
            pos_dict_size = len(pos_to_ix)
            if by_bpe or by_char:
                train_sents = [[(idxs[0], idxs[1], tag_idx)
                                for idxs, tag_idx in zip(sent, sent_tags[0])]
                               for sent, sent_tags in zip(train_sents, train_poses)]
                if val_data:
                    val_sents = [[(word_idx, char_idx, tag_idx)
                                  for (word_idx, char_idx), tag_idx in zip(sent, sent_tags[0])]
                                 for sent, sent_tags in zip(val_sents, val_poses)]
            else:
                train_sents = [[(word_idx, tag_idx)
                                for word_idx, tag_idx in zip(sent, sent_tags[0])]
                               for sent, sent_tags in zip(train_sents, train_poses)]
                if val_data:
                    val_sents = [[(word_idx,tag_idx)
                                  for word_idx, tag_idx in zip(sent, sent_tags[0])]
                                 for sent, sent_tags in zip(val_sents, val_poses)]
        field_models = [pos_model]
        for field_idx, field_name in enumerate(field_names[1:]):

            logger.info(f"Preparing {field_name} training data:")
            logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

            field_training_data = [(sent, [tag_set[field_idx+1] for tag_set in tags]) for sent, tags in training_data]

            field_tag_to_ix = prepare_tag_dict(field_training_data)
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            torch.save(field_tag_to_ix, field_dict_path)

            field_model_path = model_path_parts[0] + f"-{field_name}." + model_path_parts[1]
            if val_data:
                val_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=(field_idx+1)).to(device=device)]
                                  for (val_sent, tag_sets) in val_data]
            else:
                val_field_tags = None
            train_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx+1).to(device=device)]
                                for (train_sent, tag_sets) in training_data]

            logger.info(f"Finished preparing {field_name} data:")
            logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

            field_model = train_tag(train_sents, val_sents, train_field_tags, val_field_tags, field_model_path,
                                    word_to_ix, char_to_ix, bpe_to_ix, [field_tag_to_ix], word_emb_dim, char_emb_dim,
                                    hidden_dim, dropout, num_kernels, kernel_width, by_char, by_bpe, cnn, directions,
                                    device, save_all_models, save_best_model, epochs, lr, batch_size,
                                    weight_decay=weight_decay, pos_dict_size=pos_dict_size, seed=seed)
            field_models.append(field_model)

        return field_models[0], field_models[1], field_models[2], field_models[3], field_models[4]

    else:
        return pos_model


def train_tag(train_sents, val_sents, train_tags, val_tags, model_path, word_to_ix, char_to_ix,
              bpe_to_ix, tag_to_ix_list, word_emb_dim, char_emb_dim,
              hidden_dim, dropout, num_kernels=1000, kernel_width=6, by_char=False,
              by_bpe=False, cnn=False, directions=1, device='cpu',
              save_all_models=False, save_best_model=True, epochs=300, lr=0.1,
              batch_size=8, weight_decay=0, pos_dict_size=0, loss_weights=None, seed=42):
    """
    This is the central function that runs the training process; it trains a model on a given tag (or set of tags),
    with or without early stopping, based on the hyperparameters provided to the function call. It saves and returns
    the best model.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    base_model = base_model_factory(by_char or by_bpe, cnn)

    model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_to_ix),
                       len(char_to_ix) if by_char else len(bpe_to_ix), [len(tag_to_ix) for tag_to_ix in tag_to_ix_list],
                       num_kernels, kernel_width, directions=directions, device=device, model_type=base_model,
                       pos_dict_size=pos_dict_size)

    # move to gpu if supported
    model = model.to(device=device)

    loss_function = nn.NLLLoss().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info("Begin training:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    best_score = math.inf
    best_model = None
    best_model_path = None
    patience = 0
    val_loss = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        if epoch % 10 == 0:
            logger.info("Beginning epoch {}:".format(epoch))
            logger.info(datetime.datetime.now().strftime("%H:%M:%S"))
            sys.stdout.flush()
        count = 0
        r = list(range(len(train_sents)))
        random.shuffle(r)

        for i in range(math.ceil(len(train_sents)/batch_size)):
            batch = r[i*batch_size:(i+1)*batch_size]
            losses = []
            for j in batch:

                sentence = train_sents[j]
                tags = train_tags[j]

                # skip sentences with zero words or zero tags (though it should be equivalent)
                if (not len(sentence)) or (not len(tags)):
                    continue
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden(hidden_dim)

                sentence_in = sentence
                targets = tags

                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                loss = [loss_function(tag_scores[i], targets[i]) for i in range(len(tag_scores))]
                loss = torch.stack(loss)
                if loss_weights:
                    if len(loss_weights) != len(loss):
                        logger.info(f"Received {len(loss_weights)} weights, for {len(loss)} tasks. Using equal weights.")
                        avg_loss = sum(loss)/len(loss)
                    else:
                        weighted_loss_sum = 0
                        for task_loss, weight in zip(loss, loss_weights):
                            weighted_loss_sum += task_loss*weight
                        avg_loss = weighted_loss_sum/sum(loss_weights)

                else:
                    avg_loss = sum(loss)/len(loss)
                losses.append(avg_loss)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()

            losses = torch.stack(losses)
            total_loss = sum(losses)/len(losses)  # average over all sentences in batch
            total_loss.backward()
            running_loss += total_loss.item()
            optimizer.step()
            count += 1

        if patience == 5 and best_model:
            if save_best_model:
                logger.info("Saving best model at {}".format(model_path))
                torch.save(best_model.state_dict(), model_path)
                logger.info("Best validation loss: {}".format(best_score))
                sys.stdout.flush()
            break

        predicted_train = predict_tags(model, train_sents)
        logger.info("Loss and accuracy at epoch {}:".format(epoch))
        logger.info("Loss on training data: {}".format(running_loss/count))
        if val_sents:
            predicted_val = predict_tags(model, val_sents)

            val_loss = get_loss_on_val(val_sents, val_tags, predicted_val, loss_weights)

            logger.info("Loss on validation data: {}".format(val_loss))
            val_accuracy = calculate_accuracy(predicted_val, val_tags)
            logger.info("Accuracy on validation data: {}".format(val_accuracy))

        train_accuracy = calculate_accuracy(predicted_train, train_tags)

        logger.info("Accuracy on training data: {}".format(train_accuracy))

        save_path = model_path.split(".")
        save_path = save_path[0] + "_epoch_" + str(epoch + 1) + "." + save_path[1]
        if val_sents:
            if val_loss < best_score:

                base_model = base_model_factory(by_char or by_bpe, cnn)

                best_model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout,
                                        len(word_to_ix), len(char_to_ix) if by_char else len(bpe_to_ix),
                                        [len(tag_to_ix) for tag_to_ix in tag_to_ix_list], num_kernels,
                                        kernel_width, directions=directions, device=device, model_type=base_model,
                                        pos_dict_size=pos_dict_size)

                best_model.load_state_dict(model.state_dict())
                best_model_path = save_path
                best_score = val_loss
                best_accuracy = val_accuracy
                patience = 0
            else:
                patience += 1
            logger.info("Patience: {}".format(patience))
        if save_all_models:
            logger.info("Saving model at checkpoint.")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss/count,
                'val_loss': val_loss
            }, save_path)

        if epoch == epochs - 1 and best_model and best_model_path and save_best_model:
            logger.info("Reached max epochs")
            logger.info("Saving best model at {}".format(model_path))
            torch.save(best_model.state_dict(), model_path)
            if val_sents:
                logger.info("Best validation loss: {}".format(best_score))
                logger.info("Best validation accuracy: {}".format(best_accuracy))
            break

            sys.stdout.flush()
    logger.info("Finished training:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    if not best_model_path:
        logger.info("We never found a best model, saving final model")
        torch.save(model.state_dict(), model_path)

    return best_model


def prepare_data_for_mtl(field_names, training_data, val_data, device, dict_path_parts, test=False):
    all_field_dicts = []
    all_train_field_tags_misordered = []
    all_val_field_tags_misordered = []
    for field_idx, field_name in enumerate(field_names):

        logger.info(f"Preparing {field_name} data:")
        logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

        field_training_data = [(sent, [tag_set[field_idx] for tag_set in tags]) for sent, tags in training_data]

        if not test:
            field_tag_to_ix = prepare_tag_dict(field_training_data)
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            torch.save(field_tag_to_ix, field_dict_path)
            all_field_dicts.append(field_tag_to_ix)
        else:
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            field_tag_to_ix = torch.load(field_dict_path)

        if val_data:
            val_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                              for (val_sent, tag_sets) in val_data]
        else:
            val_field_tags = None
        train_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                            for (train_sent, tag_sets) in training_data]

        all_train_field_tags_misordered.append(train_field_tags)
        all_val_field_tags_misordered.append(val_field_tags)
    all_train_field_tags = reorder_sent_tags(all_train_field_tags_misordered, device)
    if val_data:
        all_val_field_tags = reorder_sent_tags(all_val_field_tags_misordered, device)
    else:
        all_val_field_tags = None

    return all_train_field_tags, all_val_field_tags, all_field_dicts


def reorder_sent_tags(misordered_tags, device):
    """
    This function receives a list of lists of tags of the following shape:
    [[field_tags], [field_tags], [field_tags], [field_tags], [field_tags]] (of length num_fields)
    where each list [field_tags] = [[sent], [sent], [sent]...]
    and each [sent] = [word_tag, word_tag, word_tag...]

    It outputs a list of length num_sentences, where each sentence is:
    sent = [(w1_t1, w2_t1, w3_t1), (w1_t2, w2_t2, w3_t2)....] so len(sent) == num_fields
    :param misordered_tags:
    :return:
    """
    num_fields = len(misordered_tags)

    ordered_sents = []
    for sent_idx, sent in enumerate(misordered_tags[0]):
        new_sent = []
        for field_idx in range(num_fields):
            new_sent.append([misordered_tags[field_idx][sent_idx][0][word_idx] for word_idx in range(len(sent[0]))])
        ordered_sents.append(torch.LongTensor(new_sent).to(device=device))
    return ordered_sents


def prepare_tag_dict(training_data):
    tag_to_ix = {}
    for sent, tags in training_data:
        for word_bpe, tag in zip(sent, tags):
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    if UNKNOWN not in tag_to_ix:
        tag_to_ix[UNKNOWN] = len(tag_to_ix)

    return tag_to_ix


def prepare_dictionaries(training_data, with_smoothing, frequencies):
    word_to_ix = {}
    char_to_ix = {}
    bpe_to_ix = {}
    tag_to_ix = {}

    for sent, tags in training_data:
        for word_bpe, tag in zip(sent, tags):
            word = word_bpe[0]
            bpes = word_bpe[1]
            if word not in word_to_ix:
                if not (with_smoothing and frequencies[word] <= THRESHOLD):
                    word_to_ix[word] = len(word_to_ix)
            for i in range(len(word) - 1):
                curr_char = word[i]
                next_char = word[i + 1]
                if curr_char == "'":
                    continue
                if next_char == "'":  # treat letters followed by ' as single character
                    char = curr_char + next_char
                else:
                    char = curr_char
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
            if word[-1] != "'":
                if word[-1] not in char_to_ix:
                    char_to_ix[word[-1]] = len(char_to_ix)
            if bpes:
                for bpe in bpes:
                    if bpe not in bpe_to_ix:
                        bpe_to_ix[bpe] = len(bpe_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    if UNKNOWN not in char_to_ix:
        char_to_ix[UNKNOWN] = len(char_to_ix)

    if UNKNOWN not in word_to_ix:
        word_to_ix[UNKNOWN] = len(word_to_ix)

    if UNKNOWN not in bpe_to_ix:
        bpe_to_ix[UNKNOWN] = len(bpe_to_ix)

    if UNKNOWN not in tag_to_ix:
        tag_to_ix[UNKNOWN] = len(tag_to_ix)

    return word_to_ix, char_to_ix, bpe_to_ix, tag_to_ix


def base_model_factory(by_char, cnn):
    if not by_char:
        return LSTMTagger
    else:
        if cnn:
            return CharCNNTagger
        else:
            return CharLSTMTagger


def get_loss_on_val(val_sents, val_tags, predicted_tags, loss_weights):
    """
    For giving different tasks different weights (such as giving POS a higher weight than enc)
    :param val_sents:
    :param val_tags:
    :param predicted_tags:
    :param loss_weights:
    :return:
    """
    loss_function = nn.NLLLoss()
    loss = 0.0
    divisor = len(val_sents)
    for tags, predicted in zip(val_tags, predicted_tags):
        if len(predicted) == 0:
            divisor -= 1
            continue
        sent_loss = 0.0
        if not loss_weights:
            loss_weights = [1]*len(predicted)
        for task_pred, task_tags, weight in zip(predicted, tags, loss_weights):
            sent_loss += weight*loss_function(task_pred, task_tags)
        avg_sent_loss = sent_loss/sum(loss_weights)
        loss += avg_sent_loss
    return loss/divisor


def predict_tags(model, sentences):
    """
    Predicting tags for a given model and a given set of sentences.
    In case the sentence is empty, append an empty list to the predicted tags.
    Upon receiving an invalid sentence, raise an error.
    """
    model = model.eval()
    predicted = []
    invalid_line_counter = 0
    
    for sentence in sentences:
        
        try:
            if len(sentence) == 0:
                predicted.append([])
            else:
                predicted_tags = model(sentence)
                predicted.append(predicted_tags)

        except RuntimeError as e:
            if 'zero batch' in str(e):
                predicted.append([])
                invalid_line_counter += 1
            else:
                print("An invalid sentence was given. Check input in cleaning stage.")
                raise e
            
    if invalid_line_counter != 0:
        print(f"{invalid_line_counter} invalid sentences found in the given input.")
    
    return predicted


def calculate_accuracy(predicted_tag_scores, true_tags_2d):
    """
    Shape of predicted_tag_scores: []
        len(predicted_tag_scores) = len(sentences)
        len(predicted_tag_scores[i]) = num_fields (this is a tensor)
        shape(predicted_tag_scores[j]) = torch.size(num_tags_in_field_j, num_words_in_sentence)
    Therefore shape of results:
        len(results) = len(sentences)*num_fields, with [[tags of sent_1,field_1], [tags of sent_1,field_2], ...]
    And shape of true_tags is the same as results
    :param predicted_tag_scores:
    :param true_tags_2d:
    :return:
    """

    score = 0
    results = [[np.argmax(word_scores.cpu().detach().numpy()) for word_scores in field_scores]
               for sent_scores in predicted_tag_scores for field_scores in sent_scores]
    true_tags = [tag for tags in true_tags_2d for tag in tags]
    num_tags = 0
    for sent_result, sent_true in zip(results, true_tags):
        sent_true = sent_true.cpu()
        for result, true in zip(np.array(sent_result).flatten(), np.array(sent_true).flatten()):
            if result == true:
                score += 1
            num_tags += 1
    return score/num_tags


def get_pos_from_idxs_path(pos_idxs, pos_dict_path):
    pos_dict = torch.load(pos_dict_path)
    ix_to_tag = reverse_dict(pos_dict)
    literal_pos_tags = [[ix_to_tag.get(tag, 'OOV') for tag in sentence] for sentence in pos_idxs]
    return literal_pos_tags


def test(test_data, model_path, word_dict_path, char_dict_path, bpe_dict_path,
         tag_dict_path, word_emb_dim, char_emb_dim, hidden_dim, dropout,
         num_kernels, kernel_width, by_char=False, by_bpe=False, out_path=None,
         cnn=False, directions=1, device='cpu', morph=False, use_true_pos=False,
         test_sent_sources=None, enforce_legal_morphology=False):
    """

    Prepares all the data, and then calls the function that actually runs testing
    """

    if not out_path:
        out_path = str(datetime.date.today())

    field_names = ["pos", "an1", "an2", "an3", "enc"]

    model_path_parts = model_path.split(".")
    dict_path_parts = tag_dict_path.split(".")

    word_to_ix = torch.load(word_dict_path)
    char_to_ix = torch.load(char_dict_path)
    bpe_to_ix = torch.load(bpe_dict_path)

    test_words = [[word[0] for word in test_sent[0]] for test_sent in test_data]

    if by_char:
        test_sents = [prepare_sequence_for_chars(test_sent[0], word_to_ix, char_to_ix)
                      for test_sent in test_data]
    elif by_bpe:
        test_sents = [prepare_sequence_for_bpes(test_sent[0], word_to_ix, bpe_to_ix)
                      for test_sent in test_data]
    else:
        test_sents = [prepare_sequence_for_words(test_sent[0], word_to_ix)
                      for test_sent in test_data]

    if morph == MTL:

        model_path = model_path

        all_test_field_tags, _, _ = prepare_data_for_mtl(field_names, test_data, None, device, dict_path_parts, test=True)
        all_field_dict_paths = [dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
                                for field_name in field_names]

        logger.info(f"Finished preparing MTL data:")
        logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

        mtl_results = test_morph_tag(None, test_sents, all_test_field_tags, None, test_words, None, model_path,
                                     word_to_ix, char_to_ix, bpe_to_ix, all_field_dict_paths,
                                     word_emb_dim, char_emb_dim, hidden_dim, dropout, num_kernels, kernel_width,
                                     enforce_legal_morphology, by_char, by_bpe, out_path, cnn, directions, device,
                                     field_names=field_names, test_sent_sources=test_sent_sources)
        return mtl_results

    results = []

    pos_model_path = model_path_parts[0] + f"-pos." + model_path_parts[1]
    pos_dict_path = dict_path_parts[0] + f"-pos." + dict_path_parts[1]
    pos_out_path = out_path + f"-pos"

    pos_tag_to_ix = torch.load(pos_dict_path)
    test_pos_tags = [[prepare_target(tag_sets, pos_tag_to_ix, field_idx=0).to(device=device)]
                       for (train_sent, tag_sets) in test_data]

    pos_tag_dictionary_reversed = {value: key for key, value in pos_tag_to_ix.items()}
    
    pos_results = test_morph_tag(pos_tag_dictionary_reversed, test_sents, test_pos_tags, None, test_words, None, pos_model_path,
                                 word_to_ix, char_to_ix,
                                 bpe_to_ix, [pos_dict_path], word_emb_dim, char_emb_dim, hidden_dim, dropout,
                                 num_kernels, kernel_width, False, by_char, by_bpe, pos_out_path, cnn, directions,
                                 device, field_names=["pos"], return_shaped_results=(morph==HIERARCHICAL),
                                 test_sent_sources=test_sent_sources)
    results.append(pos_results)

    if morph == FLAT or morph == HIERARCHICAL:

        pos_dict_size = 0

        if morph == HIERARCHICAL:
            if use_true_pos:
                test_pos = [tags[0] for tags in test_pos_tags]
            else:
                test_pos = [tags[0] for tags in pos_results[1]]
            if by_bpe or by_char:
                test_sents = [[(idxs[0], idxs[1], tag_idx) for idxs, tag_idx in zip(sent, sent_tags)]
                              for sent, sent_tags in zip(test_sents, test_pos)]
            else:
                test_sents = [[(word_idx, tag_idx) for word_idx, tag_idx in zip(sent, sent_tags)]
                              for sent, sent_tags in zip(test_sents, test_pos)]
            pos_dict_size = len(pos_tag_to_ix)
            results[0] = pos_results[0][0]
        else:
            results[0] = pos_results[0]

        for field_idx, field in enumerate(field_names[1:]):

            field_idx += 1

            field_model_path = model_path_parts[0] + f"-{field}." + model_path_parts[1]
            field_dict_path = dict_path_parts[0] + f"-{field}." + dict_path_parts[1]
            field_out_path = out_path + f"-{field}"

            field_tag_to_ix = torch.load(field_dict_path)
            test_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                               for (train_sent, tag_sets) in test_data]

            field_results = test_morph_tag(pos_tag_dictionary_reversed, test_sents, test_field_tags, test_pos_tags, test_words,
                                           field_idx, field_model_path, word_to_ix, char_to_ix, bpe_to_ix, [field_dict_path],
                                           word_emb_dim, char_emb_dim, hidden_dim, dropout, num_kernels, kernel_width,
                                           enforce_legal_morphology, by_char, by_bpe, field_out_path, cnn, directions, device,
                                           pos_dict_size, field_names=[field], test_sent_sources=test_sent_sources)
            results.append(field_results[0])

        return results

    else:
        return pos_results[0]


def test_morph_tag(pos_tag_dictionary, test_sents, test_field_tags, test_pos_tags, test_words, field_index, model_path,
                   word_dict, char_dict, bpe_dict, tag_dict_path_list, word_emb_dim, char_emb_dim, hidden_dim, dropout, num_kernels,
                   kernel_width, legal_morph, by_char=False, by_bpe=False, out_path=None, cnn=False, directions=1, device='cpu',
                   pos_dict_size=0, return_shaped_results=False, field_names=None, test_sent_sources=None):

    if not out_path:
        out_path = str(datetime.date.today())

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    # checkpoint = torch.load(load_path, map_location=map_location)

    # tag_dicts are dictionaries mapping tag to index
    tag_dict_list = [torch.load(tag_dict_path) for tag_dict_path in tag_dict_path_list]

    base_model = base_model_factory(by_char or by_bpe, cnn)

    model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                       len(char_dict) if by_char else len(bpe_dict), [len(tag_dict) for tag_dict in tag_dict_list],
                       num_kernels, kernel_width, directions=directions, device=device, pos_dict_size=pos_dict_size,
                       model_type=base_model)

    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model = model.to(device=device)

    tag_scores = predict_tags(model, test_sents)
    if return_shaped_results:
        # shape of tag_scores: [first sentence:[[(all tag scores for w1_f1 - max is the score you want),
        #                                        (w2_f1)], [(w1_f2), (w2_f2)...]...],
        #                      [second sentence: [[(w1_f1), (w2_f1), (w3_f1)], [(w1_f2), (w2_f2), (w3_f2)]]]
        shaped_results = [[[np.argmax(word_scores.cpu().detach().numpy()) for word_scores in field_scores]
                           for field_scores in sentence_scores]
                          for sentence_scores in tag_scores]

    ix_to_tag_list = [reverse_dict(tag_dict) for tag_dict in tag_dict_list]
    literal_test_tags = []
    for sentence in test_field_tags:
        sent_literal = []
        for field_idx, field_tags in enumerate(sentence):
            field_literal = [ix_to_tag_list[field_idx].get(tag.item(), 'OOV') for tag in field_tags]
            sent_literal.append(field_literal)
        literal_test_tags.append(sent_literal)

    if legal_morph:

        index_to_tag_list = [pos_tag_dictionary, ix_to_tag_list[0]]

        test_pos = [tensor for sublist in test_pos_tags
                    if isinstance(sublist, list) and len(sublist) == 1 and isinstance(sublist[0], torch.Tensor)
                    for tensor in sublist]

        write_predictions_to_file(ix_to_tag_list, test_pos_tags, pos_tag_dictionary, field_index, test_sents,
                                  test_words, test_pos, tag_scores, out_path + "-tagged.tsv", ix_to_tag_list, word_dict,
                                  legal_morph, ground_truth=literal_test_tags, field_names=field_names,
                                  test_sent_sources=test_sent_sources)

        results = []
        for sent_idx, sentence_scores in enumerate(tag_scores):
            sentence_results = []
            for _, field_scores in enumerate(sentence_scores, 1):
                field_results = []
                for word_idx, word_scores in enumerate(field_scores):
                    # apply filter to word scores according to analysis type
                    filter_invalid_analysis_tags(index_to_tag_list, test_pos, word_scores, field_results,
                                                 field_index, sent_idx, word_idx)

                sentence_results.append(field_results)
            results.append(sentence_results)

    else:

        results = [[[np.argmax(word_scores.cpu().detach().numpy()) for word_scores in field_scores]
                    for field_scores in sentence_scores] for sentence_scores in tag_scores]

    literal_test_predicted = []
    for sentence in results:
        sent_literal = []
        for field_idx, field_tags in enumerate(sentence):
            try:
                field_literal = [ix_to_tag_list[field_idx].get(tag.item(), 'OOV') for tag in field_tags]
                sent_literal.append(field_literal)
            except:
                print("field_idx: ", field_idx, "\n")
                print("field_tags: ", field_tags, "\n")
                print("sent: ", sentence, "\n")
                print("ix_to_tag_list[field_idx]: ", ix_to_tag_list[field_idx], "\n")

        literal_test_predicted.append(sent_literal)
    report_dicts = get_classification_report(literal_test_tags, literal_test_predicted, out_path, model_path,
                                             len(tag_dict_list), field_names=field_names)

    if not field_names:
        field_names = [f"{i}" for i in range(len(report_dicts))]

    for field_name, report_dict in zip(field_names, report_dicts):
        logger.info(f"Result {field_name} precision: {report_dict['accuracy']}")
        logger.info(f"Result {field_name} (weighted) recall: {report_dict['weighted avg']['recall']}")
        logger.info(f"Result {field_name} (weighted) f1: {report_dict['weighted avg']['f1-score']}")
    if return_shaped_results:
        return report_dicts, shaped_results
    return report_dicts


def get_classification_report(test_tags, test_predicted, out_path, model_path, num_fields, field_names=None):
    if not field_names:
        field_names = [f"{i}" for i in range(num_fields)]

    outpaths = [out_path+f"-{field_name}.report" for field_name in field_names]
    report_dicts = []
    for field_idx, out_path in enumerate(outpaths):
        field_true = [tag for sent in test_tags for tag in sent[field_idx]]
        
        field_predicted = []
        for sentence in test_predicted:
            if not sentence:
                continue
            else:
                for tag in sentence[field_idx]:
                    field_predicted.append(tag)

        filtered_field_lists = [(value_true, value_predicted) for value_true, value_predicted in
                          zip(field_true, field_predicted) if value_true != 'NA']
        filtered_true, filtered_predicted = zip(*filtered_field_lists)

        report = classification_report(filtered_true, filtered_predicted)
        report_dict = classification_report(filtered_true, filtered_predicted, output_dict=True)
        report_dicts.append(report_dict)
        with open(out_path, 'w+', encoding='utf8') as report_file:
            report_file.write("Classification report:\n")
            report_file.write("Model: {}\n".format(model_path))
            report_file.write("-------------------------------------\n")
            report_file.write(report)
    return report_dicts


def write_predictions_to_file(ix_to_tag_list, test_pos_tags, pos_tag_dictionary, field_index, sentences, test_words,
                              test_pos, tag_scores, out_path, tag_dict_list, word_dict, legal_morph,
                              ground_truth=None, field_names=None, test_sent_sources=None):
    if legal_morph:

        test_pos = [tensor for sublist in test_pos_tags
                    if isinstance(sublist, list) and len(sublist) == 1 and isinstance(sublist[0], torch.Tensor)
                    for tensor in sublist]

        index_to_tag_list = [pos_tag_dictionary, ix_to_tag_list[0]]

        results = []
        for sent_idx, sentence_scores in enumerate(tag_scores):
            sentence_results = []
            for _, field_scores in enumerate(sentence_scores, 1):
                field_results = []
                for word_idx, word_scores in enumerate(field_scores):
                    # apply filter to word scores according to analysis type
                    filter_invalid_analysis_tags(index_to_tag_list, test_pos, word_scores, field_results,
                                                 field_index, sent_idx, word_idx)

                sentence_results.append(field_results)
            results.append(sentence_results)

    else:

        results = [[[np.argmax(word_scores.cpu().detach().numpy()) for word_scores in field_scores]
                    for field_scores in sentence_scores] for sentence_scores in tag_scores]

    if not test_sent_sources:
        test_sent_sources = [("", "") for _ in sentences]
    num_fields = len(tag_dict_list)
    with open(out_path, 'w+', encoding='utf8', newline="") as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        if ground_truth:
            column_names = ['source_file', 'source_sheet', 'sentence_id', 'word', 'true_pos']
            if field_names:
                if len(field_names) == num_fields:
                    for name in field_names:
                        column_names.extend([f"true_{name}", f"predicted_{name}"])
                else:
                    logger.info("Please provide field names according to number of fields")
            else:
                for i in range(num_fields):
                    column_names.extend([f"true_{i}", f"predicted_{i}"])
            tsv_writer.writerow(column_names)
            for i, (sent_words, true_fields, pred_fields, source, true_pos) in enumerate(zip(test_words, ground_truth,
                                                                                             results, test_sent_sources,
                                                                                             test_pos)):
                for j, word in enumerate(sent_words):
                    row = [source[0], source[1], i, word, pos_tag_dictionary[np.int64(true_pos[j].cpu())]]
                    for field_idx, (field_true_tags, field_pred_tags) in enumerate(zip(true_fields, pred_fields)):
                        row.extend([field_true_tags[j], tag_dict_list[field_idx][field_pred_tags[j]]])
                    tsv_writer.writerow(row)
        else:
            column_names = ['source_file', 'source_sheet', 'sentence_id', 'word']
            if field_names:
                if len(field_names) == num_fields:
                    for name in field_names:
                        column_names.extend([f"predicted_{name}"])
                else:
                    logger.info("Please provide field names according to number of fields")
            else:
                for i in range(num_fields):
                    column_names.extend([f"predicted_{i}"])
            tsv_writer.writerow(column_names)
            for i, (sentence, pred_fields, source) in enumerate(zip(sentences, results, test_sent_sources)):
                for j, word in enumerate(sentence):
                    row = [source[0], source[1], i, word[0]]
                    for field_idx, field_pred_tags in enumerate(pred_fields):
                        row.extend([tag_dict_list[field_idx][field_pred_tags[j]]])
                    tsv_writer.writerow(row)

def filter_invalid_pos_tags(ix_to_tag_list, field_results, untagged_sents, word_scores, sent_idx, word_idx):
    """
    Allows filtering cases words are tagged with underscore ("_") or numerical cases.
    In cases that the best POS tag chosen for a word is an underscore and is a legal word,
    the second-best part of speech tag is chosen from the POS score list for that word.
    ix_to_tag_list[0] is the list of all possible POS tags for a word.
    All non-numerical words that are at least 3 letters long are compelled to receive a valid POS,
    meaning one which is not an underscore.
    """

    if ix_to_tag_list[0][np.argmax(word_scores.cpu().detach().numpy())] != "_":
        field_results.append(np.argmax(word_scores.cpu().detach().numpy()))

    else:
        word_text = untagged_sents[sent_idx][word_idx][0].strip()
        if word_text.isdigit() or len(word_text) < 3:
            field_results.append(np.argmax(word_scores.cpu().detach().numpy()))
        else:
            sorted_indices = np.argsort(word_scores.cpu().detach().numpy())
            field_results.append(sorted_indices[1])


def filter_invalid_analysis_tags(ix_to_tag_list, test_pos, word_scores, field_results, field_idx, sent_idx, word_idx):
    """
    This function oversees the applying of legal morphological values, according to every field (analysis type),
    conforming to legal_values.json.
    By applying a mask to word_scores, the best tag is chosen only from the group of legal tags.
    indices of illegal values are neutralized using -np.inf while the latter are set to zero.

    :param ix_to_tag_list: a dictionary containing all the sets of tags according to every analysis field.
                           the first field in index 0 is the pos tags.
    :param test_pos: a list of part of speech tags.
    :param word_scores: scores per each word.
    :param field_results: results by field.
    :param field_idx: the morphological analysis type.
    :param sent_idx: number of the sentence object.
    :param word_idx: the index of the word within the current sentence.
    """
    all_field_tags = ix_to_tag_list[1] if len(ix_to_tag_list) == 2 else ix_to_tag_list[field_idx]

    analyses_dict = {1: 'analysis1', 2: 'analysis2', 3: 'analysis3'}
    pos_index = np.int64(test_pos[sent_idx][word_idx].cpu().detach().numpy()) if isinstance(
        test_pos[sent_idx][word_idx], torch.Tensor) else test_pos[sent_idx][word_idx]

    pos_value = ix_to_tag_list[0][pos_index]

    legal_morph_values = (legal_morph_values['pos'][pos_value][analyses_dict[field_idx]]
                          if (pos_value in legal_morph_values['pos']) and field_idx < 4
                          else legal_morph_values['enclitic'])

    mask = np.where(np.array([all_field_tags.get(index, None) in legal_morph_values
                              for index in range(len(all_field_tags))]), 0, -np.inf)

    field_results.append(
        # append scores with or without applying mask, according to POS value
        np.argmax(word_scores.cpu().detach().numpy() + mask)
        if (pos_value in legal_morph_values['pos'])
        else np.argmax(word_scores.cpu().detach().numpy()))


def tag(data_path, model_path, word_dict_path, char_dict_path,
        bpe_dict_path, tag_dict_path, word_emb_dim, char_emb_dim, hidden_dim, dropout,
        num_kernels, kernel_width, by_char=False, by_bpe=False,
        out_path=None, cnn=False, directions=1, device='cpu', morph=None, use_true_pos=False,
        legal_morph=False):
    """

    :param data_path:
    :param model_path:
    :param word_dict_path:
    :param char_dict_path:
    :param bpe_dict_path:
    :param tag_dict_path:
    :param word_emb_dim:
    :param char_emb_dim:
    :param hidden_dim:
    :param dropout:
    :param num_kernels:
    :param kernel_width:
    :param by_char:
    :param by_bpe:
    :param out_path:
    :param cnn:
    :param directions:
    :param device:
    :param morph:
    :param use_true_pos:
    :return:
    """
    # This is the function for actually just tagging
    untagged_data, untagged_sent_objects = load_data.prepare_untagged_data(data_path)
    untagged_sents = [sent for sent in untagged_data if len(sent) > 0]
    if len(untagged_sents) != len(untagged_sent_objects):
        print("Length of sentences not the same!!!!!!")

    model_path_parts = model_path.split(".")
    dict_path_parts = tag_dict_path.split(".")
    if morph:
        field_names = ["pos", "an1", "an2", "an3", "enc"]
    else:
        field_names = ["pos"]

    if device == torch.device('cpu'):
        map_location = device
    else:
        map_location = None

    word_dict = torch.load(word_dict_path)
    char_dict = torch.load(char_dict_path)
    bpe_dict = torch.load(bpe_dict_path)
    # tag_dict is a dictionary mapping tag to index!
    tag_dict_path_list = [dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
                            for field_name in field_names]
    tag_dict_list = [torch.load(tag_dict_path) for tag_dict_path in tag_dict_path_list]
    ix_to_tag_list = [reverse_dict(tag_dict) for tag_dict in tag_dict_list]

    base_model = base_model_factory(by_char or by_bpe, cnn)

    if by_char:
        test_words = [prepare_sequence_for_chars(sent, word_dict, char_dict) for sent in untagged_sents]
    elif by_bpe:
        test_words = [prepare_sequence_for_bpes(sent, word_dict, bpe_dict) for sent in untagged_sents]
    else:
        test_words = [prepare_sequence_for_words(sent, word_dict).to(device=device) for sent in untagged_sents]

    if morph == MTL:
        model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                           len(char_dict) if by_char else len(bpe_dict), [len(tag_dict) for tag_dict in tag_dict_list],
                           num_kernels, kernel_width, directions=directions, device=device,
                           model_type=base_model)
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model = model.to(device=device)
        tag_scores = predict_tags(model, test_words)
        results = [[[np.argmax(word_scores.cpu().detach().numpy()) for word_scores in field_scores]
                     for field_scores in sentence_scores] for sentence_scores in tag_scores]

    else:
        results_by_field = []
        pos_model_path = model_path_parts[0] + f"-pos." + model_path_parts[1]
        pos_dict = tag_dict_list[0]
        model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                           len(char_dict) if by_char else len(bpe_dict), [len(pos_dict)],
                           num_kernels, kernel_width, directions=directions, device=device,
                           model_type=base_model)

        model.load_state_dict(torch.load(pos_model_path, map_location=map_location))
        model = model.to(device=device)

        tag_scores = predict_tags(model, test_words)

        for sent_idx, sentence_scores in enumerate(tag_scores):
            sentence_results = []
            for _, field_scores in enumerate(sentence_scores):
                field_results = []
                for word_idx, word_scores in enumerate(field_scores):
                    filter_invalid_pos_tags(ix_to_tag_list, field_results, untagged_sents,
                                            word_scores, sent_idx, word_idx)
                sentence_results.append(field_results)
            results_by_field.append(sentence_results)
        results_by_field = [results_by_field]

        if morph == FLAT or morph == HIERARCHICAL:

            pos_dict_size = 0

            if morph == HIERARCHICAL:
                if use_true_pos:
                    # Be very sure the words have POS tags;
                    # otherwise you'll be using a whole lot of Nones for prediction
                    test_pos = [[word_an.pos for word_an in sent_obj.word_analyses]
                                for sent_obj in untagged_sent_objects]
                    test_pos = [torch.LongTensor([get_index(pos, pos_dict) for pos in sent_poses]).to(device=device) for sent_poses in test_pos]
                else:
                    test_pos = []
                    for sent in results_by_field[0]:
                        if sent:
                            test_pos.append(sent[0])

                if by_bpe or by_char:
                    test_words = [[(idxs[0], idxs[1], tag_idx) for idxs, tag_idx in zip(sent, sent_tags)]
                                  for sent, sent_tags in zip(test_words, test_pos)]
                else:
                    test_words = [[(word_idx, tag_idx) for word_idx, tag_idx in zip(sent, sent_tags)]
                                  for sent, sent_tags in zip(test_words, test_pos)]
                pos_dict_size = len(tag_dict_list[0])

            for field_idx, field in enumerate(field_names[1:]):
                results_by_field.append([])
                field_idx += 1

                field_model_path = model_path_parts[0] + f"-{field}." + model_path_parts[1]
                model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                                   len(char_dict) if by_char else len(bpe_dict), [len(tag_dict_list[field_idx])],
                                   num_kernels, kernel_width, directions=directions, device=device,
                                   model_type=base_model, pos_dict_size=pos_dict_size)

                model.load_state_dict(torch.load(field_model_path, map_location=map_location))
                model = model.to(device=device)

                tag_scores = predict_tags(model, test_words)

                results_by_field = []

                # filter tag scores according to legal morph. values
                if legal_morph:

                    results_by_field = [[] for i in enumerate(field_names)]
                    for sent_idx, sentence_scores in enumerate(tag_scores):
                        sentence_results = []
                        for _, field_scores in enumerate(sentence_scores, 1):
                            field_results = []
                            for word_idx, word_scores in enumerate(field_scores):
                                # apply filter to word scores according to analysis type
                                filter_invalid_analysis_tags(ix_to_tag_list, test_pos, word_scores, field_results,
                                                             field_idx, sent_idx, word_idx)

                            sentence_results.append(field_results)
                        results_by_field[field_idx].append(sentence_results)

                else:
                    # Not enforcing legal morphological values
                    results_by_field.append([[[np.argmax(word_scores.cpu().detach().numpy())
                                               for word_scores in field_scores]
                                              for field_scores in sentence_scores]
                                             for sentence_scores in tag_scores])

            results = reshape_by_field_to_by_sent(results_by_field)

        else:
            # this is POS only tagging
            results = reshape_by_field_to_by_sent(results_by_field, num_fields=1)  # TODO make sure this is correct

    updated_sentences = add_tags_to_sent_objs(untagged_sent_objects, results, ix_to_tag_list, field_names)
    write_tagged_sents(updated_sentences, out_path)


def reshape_by_field_to_by_sent(results_by_field, num_fields=5):
    """
    Input is list of length num_fields, where each inner list is of sentences, with num words
    Output needs to be list of length num_sentences, where len(output[i]) == num_fields
    :param results_by_field:
    :param num_fields:
    :return:
    """
    sentences = []
    
    for sentence_idx in range(len(results_by_field[0])):
        sentence_words = []
    
        for field_idx in range(num_fields):
            
            sentence = results_by_field[field_idx][sentence_idx]
            if not sentence:
                # in case current sentence is empty, append empty list
                sentence_words.append([])
            else:
                sentence_words.append(sentence[0])
        
        sentences.append(sentence_words)
    
    return sentences


def add_tags_to_sent_objs(sentences, tags, ix_to_tag_list, field_names):
    """
    Shape of tags:
    len(tags) == len(sentences)
    len(tags[i]) == len(field_names) == len(ix_to_tag_list)
    len(tags[i][j]) == len(field_names)
    :param sentences:
    :param tags:
    :param ix_to_tag_list:
    :param field_names:
    :return:
    """
    field_to_column_dict = {"pos": "pos", "an1": "analysis1", "an2": "analysis2",
                            "an3": "analysis3", "enc": "enclitic_pronoun"}
    for sentence, sent_tags in zip(sentences, tags):
        for field_name, field_tags, ix_to_tag in zip(field_names, sent_tags, ix_to_tag_list):
            for word_an, tag in zip(sentence.word_analyses, field_tags):
                word_an.set_val(field_to_column_dict[field_name], ix_to_tag[tag])
    return sentences


def write_tagged_sents(sentences, dir):
    write_sentences_to_excel(sentences, dir)


def kfold_val(data_paths, model_path, word_dict_path, char_dict_path, bpe_path,
              tag_dict_path, result_path, k, word_emb, char_emb, hidden_dim,
              dropout, num_kernels=1000, kernel_width=6, by_char=False, by_bpe=False,
              with_smoothing=False, cnn=False, directions=1, device='cpu',
              epochs=300, morph=None, weight_decay=0, use_true_pos=False, loss_weights=(1,1,1,1,1)):
    logger.info("Beginning k-fold validation")
    results = []
    fold = 0
    for train_sentences, val_sentences, test_sentences, train_word_count in \
            load_data.prepare_kfold_data(data_paths, k=k, seed=0):
        logger.info("Beginning fold #{}".format(fold+1))
        new_model_path = add_fold_dir_to_path(model_path, fold)
        new_word_path = add_fold_dir_to_path(word_dict_path, fold)
        new_char_path = add_fold_dir_to_path(char_dict_path, fold)
        new_tag_path = add_fold_dir_to_path(tag_dict_path, fold)
        new_bpe_path = add_fold_dir_to_path(bpe_path, fold)

        train(train_sentences, val_sentences, new_model_path, new_word_path,
              new_char_path, new_bpe_path, new_tag_path, train_word_count, word_emb,
              char_emb, hidden_dim, dropout, num_kernels, kernel_width, by_char, by_bpe,
              with_smoothing, cnn, directions, device, epochs=epochs, morph=morph,
              weight_decay=weight_decay, loss_weights=loss_weights)

        new_result_path = add_fold_dir_to_path(result_path, fold)
        results.append(test(test_sentences, new_model_path, new_word_path, new_char_path, new_bpe_path,
                            new_tag_path, word_emb, char_emb, hidden_dim, dropout, num_kernels,
                            kernel_width, by_char, by_bpe, out_path=new_result_path,
                            cnn=cnn, directions=directions, device=device, morph=morph, use_true_pos=use_true_pos))

        fold += 1
    agg_result_path = result_path + ".agg_res"
    mic_prec = []
    mac_prec = []
    weight_prec = []
    with open(agg_result_path, 'w+') as f:
        if not morph:
            for i, res in enumerate(results):
                micro = res.get("micro avg", res["accuracy"])
                f.write("Fold {}:\nmicro avg: {}\nmacro avg: {}\nweighted avg: {}\n".format(
                    i, micro, res["macro avg"], res["weighted avg"]))
                if "micro avg" in res:
                    mic_prec.append(res["micro avg"]["precision"])
                else:
                    mic_prec.append(res["accuracy"])  # TODO you need to check how this appears in the dict!
                mac_prec.append(res["macro avg"]["precision"])
                weight_prec.append(res["weighted avg"]["precision"])
            avg = np.mean(mic_prec)
            std = np.std(mic_prec)
            f.write("------------\nMicro:\nAverage precision: {}\nStandard deviation:{}\n".format(avg, std))
            avg = np.mean(mac_prec)
            std = np.std(mac_prec)
            f.write("------------\nMacro:\nAverage precision: {}\nStandard deviation:{}\n".format(avg, std))
            avg = np.mean(weight_prec)
            std = np.std(weight_prec)
            f.write("------------\nWeighted:\nAverage precision: {}\nStandard deviation:{}\n".format(avg, std))
        else:
            an_names = ["pos", "analysis1", "analysis2",
                        "analysis3", "enclitic"]
            for i, res in enumerate(results):
                # if isinstance(res[0], list):
                #     res = [an_res[0] for an_res in res]
                f.write(f"Fold {i} results:\n")
                for an_field, an_res in zip(an_names, res):
                    if isinstance(an_res, list):
                        an_res = an_res[0]
                    micro = an_res.get("micro avg", an_res["accuracy"])
                    f.write(f"{an_field}\nmicro: {micro}\nmacro: {an_res['macro avg']}"
                            f"\nweighted{an_res['weighted avg']}\n")
                try:
                    mic_prec.append([an_res["micro avg"]["precision"] for an_res in res])
                except KeyError:
                    mic_prec.append([an_res["accuracy"] for an_res in res])
                mac_prec.append([an_res["macro avg"]["precision"] for an_res in res])
                weight_prec.append([an_res["weighted avg"]["precision"] for an_res in res])
            avg = [np.mean([fold_res[i] for fold_res in mic_prec]) for i in range(len(an_names))]
            std = [np.std([fold_res[i] for fold_res in mic_prec]) for i in range(len(an_names))]
            for an_field, a, s in zip(an_names, avg, std):
                f.write("------------\nMicro {}:\nAverage precision: {}\nStandard deviation:{}\n".
                        format(an_field, a, s))
            avg = [np.mean([fold_res[i] for fold_res in mac_prec]) for i in range(len(an_names))]
            std = [np.std([fold_res[i] for fold_res in mac_prec]) for i in range(len(an_names))]
            for an_field, a, s in zip(an_names, avg, std):
                f.write("------------\nMacro {}:\nAverage precision: {}\nStandard deviation:{}\n".
                        format(an_field, a, s))
            avg = [np.mean([fold_res[i] for fold_res in weight_prec]) for i in range(len(an_names))]
            std = [np.std([fold_res[i] for fold_res in weight_prec]) for i in range(len(an_names))]
            for an_field, a, s in zip(an_names, avg, std):
                f.write("------------\nWeighted {}:\nAverage precision: {}\nStandard deviation:{}\n".
                        format(an_field, a, s))

    return results


def add_fold_dir_to_path(path, fold_num):
    split_path = os.path.split(path)
    new_dir = os.path.join(split_path[0], "fold_{}".format(fold_num))
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_path = os.path.join(new_dir, split_path[1])
    return new_path


def main(args):
    today = str(datetime.date.today())

    log_name = args.log_file if args.log_file else '{}.log'.format(today)
    split_path = os.path.split(log_name)
    if not os.path.exists(split_path[0]):
        os.mkdir(split_path[0])
    logger.addHandler(logging.FileHandler(log_name, 'w+'))

    char_based = args.char_based
    bpe_based = args.bpe_based
    smoothed = True if args.smoothed else False
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_path = os.path.join(args.model_dir, args.model_name)
    word_dict_path = os.path.join(args.model_dir, args.word_dict_name)
    char_dict_path = os.path.join(args.model_dir, args.char_dict_name)
    bpe_dict_path = os.path.join(args.model_dir, args.bpe_dict_name)
    tag_dict_path = os.path.join(args.model_dir, args.tag_dict_name)

    if args.result_path:
        rp_base_dir = os.path.split(args.result_path)[0]
        if not os.path.isdir(rp_base_dir):
            os.mkdir(rp_base_dir)

    morph = None
    if args.morph:
        if args.flat:
            morph = FLAT
        elif args.multitask:
            morph = MTL
        elif args.hierarchical:
            morph = HIERARCHICAL

    if args.train:
        if args.no_val:
            train_data, frequencies = load_data.prepare_train_data(args.data_paths)
            val_data = None
        else:
            _, _, train_path, val_path = split_train_val(args.data_paths, args.model_dir, max_words=args.max_words)
            train_data, frequencies = load_data.prepare_train_data([train_path])
            val_data, _ = load_data.prepare_test_data([val_path], sources=False)

        train(train_data, val_data, model_path, word_dict_path,
              char_dict_path, bpe_dict_path, tag_dict_path, frequencies,
              word_emb_dim=args.word_emb_dim, char_emb_dim=args.char_emb_dim,
              hidden_dim=args.hidden_dim, dropout=args.dropout, num_kernels=args.num_kernels,
              kernel_width=args.kernel_width, with_smoothing=smoothed,
              by_char=char_based, by_bpe=bpe_based, cnn=args.cnn, directions=args.directions,
              device=args.device, epochs=args.epochs, lr=args.learning_rate,
              batch_size=args.batch_size, morph=morph, loss_weights=args.loss_weights, seed=args.seed)
    elif args.test:
        test_data, sources = load_data.prepare_test_data(args.data_paths, out_dir=args.model_dir, sources=True)

        test(test_data, model_path, word_dict_path,
             char_dict_path, bpe_dict_path, tag_dict_path,
             word_emb_dim=args.word_emb_dim, char_emb_dim=args.char_emb_dim,
             hidden_dim=args.hidden_dim, dropout=args.dropout,
             num_kernels=args.num_kernels, kernel_width=args.kernel_width,
             by_char=char_based, by_bpe=bpe_based, cnn=args.cnn, directions=args.directions,
             out_path=args.result_path, device=args.device, morph=morph, use_true_pos=args.use_true_pos,
             test_sent_sources=sources, enforce_legal_morphology=args.legal_morph)

    elif args.tag:
        tag(args.data_paths, model_path, word_dict_path, char_dict_path, bpe_dict_path,
            tag_dict_path, word_emb_dim=args.word_emb_dim, char_emb_dim=args.char_emb_dim,
            hidden_dim=args.hidden_dim, dropout=args.dropout,
            num_kernels=args.num_kernels, kernel_width=args.kernel_width,
            by_char=char_based, by_bpe=bpe_based, cnn=args.cnn, directions=args.directions,
            out_path=args.result_path, device=args.device, morph=morph, use_true_pos=args.use_true_pos,
            legal_morph=args.legal_morph)
        
    elif args.kfold_validation:
        kfold_val(args.data_paths, model_path, word_dict_path,
                  char_dict_path, bpe_dict_path, tag_dict_path,
                  args.result_path, args.k, args.word_emb_dim, args.char_emb_dim,
                  args.hidden_dim, args.dropout, args.num_kernels, args.kernel_width,
                  char_based, bpe_based, cnn=args.cnn, directions=args.directions,
                  device=args.device, epochs=args.epochs, morph=morph, use_true_pos=args.use_true_pos,
                  loss_weights=args.loss_weights)
    else:
        print("Must select either train (-r), test (-e) or tag (-a)")


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Train, test, or calculate POS')
    parser.add_argument(
        'model_dir',
        help="Path to directory to save or load the model and all its necessary dicts")
    parser.add_argument(
        '--model_name', help="Name for model", default="ja.pt")
    parser.add_argument(
        '--word_dict_name', help="Name for word dictionary", default='words.dct')
    parser.add_argument(
        '--char_dict_name', help="Name for char dictionary", default="char.dct")
    parser.add_argument(
        '--bpe_dict_name', help="Name for BPE dictionary", default='bpe.dct')
    parser.add_argument(
        '--tag_dict_name', help="Name for POS tag dictionary", default='tags.dct')
    parser.add_argument(
        '--no_val',
        action='store_true',
        help="For training a model on all the data, without early stopping")

    parser.add_argument('data_paths', nargs='+', help="Path to data pickles to train on/test on/tag")

    parser.add_argument('--morph', action='store_true', help="Tag morphological analyses")

    morph_group = parser.add_mutually_exclusive_group()

    morph_group.add_argument(
        '--flat',
        action='store_true',
        help="Train separate model for each analysis field")
    morph_group.add_argument(
        '--multitask',
        action='store_true',
        help="Train single multitask model for POS and all morphological analyses")
    morph_group.add_argument(
        '--hierarchical',
        action='store_true',
        help="Train hierarchical models for morphological analyses")

    parser.add_argument('-we', '--word_emb_dim', type=int, default=100, help="Word embedding dimensionality")
    parser.add_argument('-ce', '--char_emb_dim', type=int, default=25,
                        help="Character (or BPE) embedding dimensionality")
    parser.add_argument('-he', '--hidden_dim', type=int, default=100, help="Hidden state dimensionality")

    parser.add_argument('--cnn', action='store_true', help="Use this flag to use char CNN instead of char LSTM")
    parser.add_argument('-nk', '--num_kernels', type=int, default=500, help="Number of kernels to use for char-CNN")
    parser.add_argument('-kw', '--kernel_width', type=int, default=6, help="Kernel width to use for char-CNN")

    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate in LSTM")
    parser.add_argument('--directions', type=int, default=2, choices=[1, 2], help="Number of directions in LSTM")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help="Size of batch")

    model_type = parser.add_mutually_exclusive_group(required=True)

    model_type.add_argument('-w', '--word_based', action='store_true')
    model_type.add_argument('-c', '--char_based', action='store_true')
    model_type.add_argument('-b', '--bpe_based', action='store_true')

    parser.add_argument('-s', '--smoothed', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-r', '--train', action='store_true')
    group.add_argument('-e', '--test', action='store_true')
    group.add_argument('-a', '--tag', action='store_true')
    group.add_argument('-kfv', '--kfold_validation', action='store_true')

    parser.add_argument('-k', type=int, default=10, help="Number of folds for kfold validation")

    parser.add_argument('--epochs', type=int, default=30, help="Maximum number of epochs to train")

    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument('-l', '--log_file', type=str, help="Path to save log file at")

    parser.add_argument('-rp', '--result_path', type=str, default="",
                        help="Path *without extension* to save test/tag results at")
    parser.add_argument('--use_true_pos',
                        action='store_true',
                        help="When testing hierarchical models, use true pos tags and not predicted pos tags")

    parser.add_argument('--testing', action='store_true',
                        help="Prints command line arguments for this run, and exits program")
    parser.add_argument('--debug', action='store_true', help='Set logger level to debug')

    parser.add_argument('--add_path', help='Append script dir to python path')

    parser.add_argument('--loss_weights',
                        nargs=5,
                        type=int,
                        default=[1, 1, 1, 1, 1],
                        help="Weights for averaging loss in MTL leaning, in order: POS, an1, an2, an3, enc")
    parser.add_argument('--seed', type=int, default=42, help="Use specific seed for random elements in training")

    parser.add_argument('--max_words',
                        type=int,
                        default=math.inf,
                        help='For training on smaller (random) subset of input sentences')

    parser.add_argument('--legal_morph', 
                        action='store_true', 
                        help='Enforce legal morphological values according to constrains in legal_values.json')

    args, unknown = parser.parse_known_args()

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.debug:
        logger.setLevel(level=logging.DEBUG)

    if args.add_path:
        import sys
        sys.path.append(args.add_path)
        import load_data
        from utils import split_train_val
        from data_classes import write_sentences_to_excel

    if not args.testing:

        # If enforcing legal morphological values, load legal_values.json
        if args.legal_morph:
            with open("legal_values.json") as legal_values_file:
                legal_morph_values = json.load(legal_values_file)

        try:
            main(args)
        
        finally:
            if args.legal_morph:
                legal_values_file.close()


    else:
        # If testing, print args and exit
        print(args)
        
