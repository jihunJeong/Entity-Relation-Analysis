from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
from torch.optim import AdamW

import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = "bert-base-multilingual-cased"  
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  '''
  # load my model
  MODEL_NAME = args.model_dir # model dir.
  model = BertForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)
  model.eval()

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)
  
  # predict answer
  pred_answer = inference(model, test_dataset, device)
  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv('./prediction/submission.csv', index=False)
  '''
  
  KFold = 1
  log_dir="./results"
  answer = np.zeros((1000, 42))
  for idx in range(KFold):
    print(f"{idx} Fold Test start")
    model = torch.load(f'{log_dir}/model{idx}.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(f'{log_dir}/model_state_dict{idx}.pt'))  # state_dict를 불러 온 후, 모델에 저장

    checkpoint = torch.load(f'{log_dir}/all{idx}.tar')   # dict 불러오기
    model.load_state_dict(checkpoint['model'])
    optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
    model.eval()

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)
    
    # predict answer
    pred_answer = inference(model, test_dataset, device)
    for idx, num in enumerate(pred_answer):
      answer[idx][num] += 1

  output = pd.DataFrame(np.argmax(answer, axis=1), columns=['pred'])
  output.to_csv('./prediction/submission.csv', index=False)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/checkpoint-500")
  args = parser.parse_args()
  print(args)
  main(args)
  
