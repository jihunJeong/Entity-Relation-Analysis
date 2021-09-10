import pickle as pickle
import os
import pandas as pd
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():
  split = 5
  #skfold = StratifiedKFold(n_splits=split)
  kfold = KFold(n_splits=split, shuffle=True, random_state=100)
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  dataset = load_data("/opt/ml/input/data/train/train.tsv")
  # for idx, num in dataset['label'].value_counts().items():
  #   if num < split:
  #     # df1 = dataset[dataset['label'] == idx].sample(n=split-num, replace=True, random_state=100)
  #     # dataset = dataset.append(df)
  #     dataset.drop(dataset[dataset['label']==idx].index, inplace=True)
  labels = dataset['label'].values
  # tokenizing dataset
  # make dataset for pytorch.
  for fold, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    train_labels = labels[train_idx]
    tokenized_train_data = tokenized_dataset(dataset.iloc[train_idx], tokenizer)
    RE_train_dataset = RE_Dataset(tokenized_train_data, train_labels)

    dev_labels = labels[valid_idx]
    tokenized_dev_data = tokenized_dataset(dataset.iloc[valid_idx], tokenizer)
    RE_dev_dataset = RE_Dataset(tokenized_dev_data, dev_labels)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    bert_config = BertConfig.from_pretrained(MODEL_NAME)
    bert_config.num_labels = 42
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
    model.to(device)
    
    print(f"{fold} Stratified K Fold start")
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    train_loader = DataLoader(RE_train_dataset, batch_size=32)
    valid_loader = DataLoader(RE_dev_dataset, batch_size=32)
    optim = AdamW(model.parameters(), lr=1e-5)
    best_accr, EPOCHS = 0, 10

    for epoch in range(EPOCHS):
        train_loss, valid_loss = AverageMeter(), AverageMeter()

        model.train()
        for iter, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            train_labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=train_labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            train_loss.update(loss.item(), len(train_labels))
            print("\rEpoch [%3d/%3d] | Iter [%3d/%3d] | Train Loss %.4f" % (epoch+1, EPOCHS, iter+1, len(train_loader), train_loss.avg), end= '')
        
        model.eval()
        correct, total = 0, 0
        for batch in valid_loader:
          with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            valid_labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels = valid_labels)
            logits = outputs[1]
            logits = torch.argmax(logits, dim=1)
            correct += (logits==valid_labels).sum().item()
            total += len(valid_labels)
            valid_loss.update(loss.item(), len(valid_labels))

        log_dir='./results'
        accr_val = (100*correct / total)
        print("Epoch [%3d/%3d] | Valid Loss %.4f | Accuracy : %.2f" % (epoch+1, EPOCHS, valid_loss.avg, accr_val))
        if best_accr < accr_val:
          best_accr = accr_val
          if not os.path.exists(log_dir):
            os.mkdir(log_dir)

          print(f"Model Save : acc - {accr_val}")
          torch.save(model, f'{log_dir}/model{fold}.pt')  
          torch.save(model.state_dict(), 
                      f'{log_dir}/model_state_dict{fold}.pt')  
          torch.save({
                  'model': model.state_dict(),
                  'optimizer': optim.state_dict()
              }, f'{log_dir}/all{fold}.tar')

def main():
  train()

if __name__ == '__main__':
  main()
