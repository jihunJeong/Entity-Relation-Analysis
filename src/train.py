import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig


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
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
  split=5
  kfold = KFold(n_splits=split, shuffle=True, random_state=100)
  # load dataset
  dataset = load_data("/opt/ml/input/data/train/train.tsv")
  train_dataset = dataset.loc[:0.8*len(dataset)]
  dev_dataset = dataset.drop(train_dataset.index)

  #dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  print(torch.cuda.is_available())
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  # setting model hyperparameter
  bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42
  model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    do_predict=True,
    save_total_limit=3,              # number of total save model.
    save_steps=100,                 # model saving step.
    num_train_epochs=10,              # total number of training epochs
    learning_rate=1e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=300,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 100,            # evaluation step.
    label_smoothing_factor=0.5,
    dataloader_num_workers=4    
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

def main():
  train()

if __name__ == '__main__':
  main()
