#Evaluation script

import pickle as pkl
import os
import pandas as pd
import random
from generate_dataloaders import *
import transformers


from transformers import (
    BertTokenizer
)


def get_predictions(model, centroids, val_loader,criterion,current_device):
    model.eval()
    token_list = []
    cluster_assignment_list = []
    flagged_index_list = []
    original_label = []
    for i, (tokens, labels, flagged_indices) in enumerate(val_loader):
            tokens = tokens.to(current_device)
            labels = labels.to(current_device)
            flagged_indices = flagged_indices.to(current_device)
            
            # forward pass and compute loss
            sentence_embed = model(tokens,flagged_indices)
            cluster_loss, cluster_assignments = criterion(sentence_embed, centroids)
            
            # store in list
            token_list+=tokens.tolist()
            flagged_index_list+=flagged_indices.tolist()
            cluster_assignment_list+=cluster_assignments.tolist()
            original_label+=labels.tolist()
            
    return token_list, flagged_index_list, cluster_assignment_list, original_label

def get_supervised_predictions(model, val_loader, criterion, current_device):
    model.eval()
    token_list = []
    cluster_assignment_list = []
    flagged_index_list = []
    original_label = []
    for i, (tokens, labels, flagged_indices) in enumerate(val_loader):
            tokens = tokens.to(current_device)
            labels = labels.to(current_device)
            flagged_indices = flagged_indices.to(current_device)
            
            # forward pass and compute loss
            class_probabilities = model(tokens,flagged_indices)
            cluster_assignments = class_probabilities.max(1,keepdim=True)[1].view(-1)
            
            # store in list
            token_list+=tokens.tolist()
            flagged_index_list+=flagged_indices.tolist()
            cluster_assignment_list+=cluster_assignments.tolist()
            original_label+=labels.tolist()
            
    return token_list, flagged_index_list, cluster_assignment_list, original_label


def decode_predictions(token_list,index_list,cluster_assignment_list,dictionary,original_label):
    decoded_tokens = [' '.join(dictionary.decode_idx_seq(sent)) for sent in token_list]
    reviews = [decoded for decoded in decoded_tokens]
    flagged_words = [r.split()[i] for (r,i) in zip(reviews,index_list)]
    reviews = [review.split('<pad>')[0] for review in reviews]
    df_pred = pd.DataFrame({'review':reviews,'index':index_list,'flagged_word':flagged_words,\
                            'assignment':cluster_assignment_list,'original':original_label})
    
    TP_cluster = df_pred[df_pred.assignment==1]
    FP_cluster = df_pred[df_pred.assignment==0]
    
    pred_1 = df_pred[df_pred.assignment==1]
    pred_0 = df_pred[df_pred.assignment==0]
    
    pred_1_manual_TP = len(pred_1[pred_1.original == 1]) / pred_1.shape[0]
    pred_0_manual_TP = len(pred_0[pred_0.original == 1]) / pred_0.shape[0]
    
    if pred_1_manual_TP >= pred_0_manual_TP:
        TP_cluster = pred_1
        FP_cluster = pred_0
    else:
        TP_cluster = pred_0
        FP_cluster = pred_1
        TP_cluster.assignment =0
        FP_cluster.assignment = 1

    return TP_cluster, FP_cluster


def performance_analysis(TP_cluster,FP_cluster,verbose=True):
    
    if (TP_cluster.shape[0] != 0) and (FP_cluster.shape[0] != 0):
        TP_rate = len(TP_cluster[TP_cluster.original==1]) / TP_cluster.shape[0]
        FP_rate = len(TP_cluster[TP_cluster.original==0]) / TP_cluster.shape[0]
        FN_rate = len(FP_cluster[FP_cluster.original==1]) / FP_cluster.shape[0]
        TN_rate = len(FP_cluster[FP_cluster.original==0]) / FP_cluster.shape[0]
    else:
        print("Zero rows in TP_cluster!")
        return
    
    accuracy = (TP_rate + TN_rate) / (TP_rate + FP_rate + FN_rate + TN_rate)
    precision = TP_rate / (TP_rate + FP_rate)
    recall = TP_rate / (TP_rate + FN_rate)
    f1_score = (2 * precision * recall) / (precision + recall)
    
    if verbose:
        print("TP_rate:",TP_rate)
        print("FP_rate:",FP_rate)
        print("FN_rate:",FN_rate)
        print("TN_rate:",TN_rate)
        print("\n")
        
        print("Accuracy:",accuracy)
        print("Precision:",precision)
        print("Recall:",recall)
        print("F1 score:",f1_score)

    return {"TP_rate":TP_rate,
            "FP_rate":FP_rate,
            "FN_rate":FN_rate,
            "TN_rate":TN_rate,
            "Accuracy":accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1 score":f1_score}

#main fcn for printing results of non-bert models
def main(model, centroids, val_loader, criterion, data_dir, current_device, verbose=True):
	#print(criterion)
    if len(centroids) == 0:
        token_list, index_list, cluster_assignment_list, original_label = get_supervised_predictions(model, val_loader, criterion, current_device)
	   
    else:
        token_list, index_list, cluster_assignment_list, original_label = get_predictions(model, centroids, val_loader, criterion, current_device)
    print(f"Total examples in val loader: {len(index_list)}")
    print(f"Assigned to cluster 1: {sum(cluster_assignment_list)}")
    dictionary = pkl.load(open(data_dir+'dictionary.p','rb'))
    pd.set_option('max_colwidth',0)
    TP_cluster, FP_cluster = decode_predictions(token_list,index_list,cluster_assignment_list,dictionary,original_label)
    results = performance_analysis(TP_cluster,FP_cluster,verbose)
    results["val_total"] = len(index_list)
    results["assigned_1"] = sum(cluster_assignment_list)

    return TP_cluster, FP_cluster, results

#main fcn for printing results of bert model
def bert(model, centroids, val_loader, criterion, data_dir, current_device):
    #print(criterion)
    if len(centroids) == 0:
        token_list, cluster_assignment_list, original_label = get_supervised_bert_predictions(model, val_loader, criterion, current_device)
       
    else:
        token_list, cluster_assignment_list, original_label = get_bert_predictions(model, centroids, val_loader, criterion, current_device)
    print(f"Total examples in val loader: {len(token_list)}")
    print(f"Assigned to cluster 1: {sum(cluster_assignment_list)}")
    dictionary = BertTokenizer.from_pretrained('bert-base-cased')
    pd.set_option('max_colwidth',0)
    TP_cluster, FP_cluster = decode_bert_predictions(token_list,cluster_assignment_list,dictionary,original_label)
    results = performance_analysis(TP_cluster,FP_cluster)
    results["val_total"] = len(index_list)
    results["assigned_1"] = sum(cluster_assignment_list)

    return TP_cluster, FP_cluster, results

def get_bert_predictions(model, centroids, val_loader,criterion,current_device):
    model.eval()
    token_list = []
    cluster_assignment_list = []
    flagged_index_list = []
    original_label = []
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
            model.eval()
            input_ids = input_ids.to(current_device)
            attention_mask = attention_mask.to(current_device)
            token_type_ids = token_type_ids.to(current_device)
            labels = labels.to(current_device)
            
            # forward pass and compute loss
            sentence_embed,attn = model(input_ids, attention_mask, token_type_ids)
            cluster_loss, cluster_assignments = criterion(sentence_embed, centroids)
            
            # store in list
            token_list+=input_ids.tolist()
            # flagged_index_list+=flagged_indices.tolist()
            cluster_assignment_list+=cluster_assignments.tolist()
            original_label+=labels.tolist()
            
    return token_list, cluster_assignment_list, original_label

def get_supervised_bert_predictions(model, val_loader, criterion, current_device):
    model.eval()
    token_list = []
    cluster_assignment_list = []
    flagged_index_list = []
    original_label = []
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
            model.eval()
            input_ids = input_ids.to(current_device)
            attention_mask = attention_mask.to(current_device)
            token_type_ids = token_type_ids.to(current_device)
            labels = labels.to(current_device)
            
            # forward pass and compute loss
            logits,attn = model(input_ids, attention_mask, token_type_ids)

            cluster_assignments = logits.max(1,keepdim=True)[1].view(-1)
            
            # store in list
            token_list+=input_ids.tolist()
            # flagged_index_list+=flagged_indices.tolist()
            cluster_assignment_list+=cluster_assignments.tolist()
            original_label+=labels.tolist()
            
    return token_list, cluster_assignment_list, original_label

def decode_bert_predictions(token_list,cluster_assignment_list,dictionary,original_label):
    decoded_tokens = [' '.join([dictionary._convert_id_to_token(word) for word in sent]) for sent in token_list]
    reviews = [decoded for decoded in decoded_tokens]
    # flagged_words = [r.split()[i] for (r,i) in zip(reviews,index_list)]
    reviews = [review.split('[PAD]')[0] for review in reviews]
    df_pred = pd.DataFrame({'review':reviews,\
                            'assignment':cluster_assignment_list,'original':original_label})
    
    TP_cluster = df_pred[df_pred.assignment==1]
    FP_cluster = df_pred[df_pred.assignment==0]
    
    pred_1 = df_pred[df_pred.assignment==1]
    pred_0 = df_pred[df_pred.assignment==0]
    
    pred_1_manual_TP = len(pred_1[pred_1.original == 1]) / pred_1.shape[0]
    pred_0_manual_TP = len(pred_0[pred_0.original == 1]) / pred_0.shape[0]
    
    if pred_1_manual_TP >= pred_0_manual_TP:
        TP_cluster = pred_1
        FP_cluster = pred_0
    else:
        TP_cluster = pred_0
        FP_cluster = pred_1
        TP_cluster.assignment =0
        FP_cluster.assignment = 1

    return TP_cluster, FP_cluster





