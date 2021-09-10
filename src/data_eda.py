import pandas as pd
import numpy as np
import argparse
import pickle as pickle
import matplotlib.pyplot as plt

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset, label_type

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--data_dir', type=str, default="train.tsv")
  args = parser.parse_args()
  data_path = "/opt/ml/input/data/train/"+args.data_dir
  df, label_type = load_data(data_path)
  df['label'] = df['label'].replace(list(label_type.values()), list(label_type.keys()))
  # print(df['label'].value_counts())
  review_len_by_eumjeol = [len(s) for s in df['sentence']]
  print('문장 개수 : {}'.format(len(review_len_by_eumjeol)))
  print('문장 최대 길이 : {}'.format(np.max(review_len_by_eumjeol)))
  print('문장 최소 길이 : {}'.format(np.min(review_len_by_eumjeol)))
  print('문장 평균 길이 : {:.2f}'.format(np.mean(review_len_by_eumjeol)))
  print('문장 길이 표준편타 : {:.2f}'.format(np.std(review_len_by_eumjeol)))
  print('문장 중간 길이 : {}'.format(np.median(review_len_by_eumjeol)))
  print('제 1사분위 길이 : {}'.format(np.percentile(review_len_by_eumjeol, 25)))
  print('제 3사분위 길이 : {}'.format(np.percentile(review_len_by_eumjeol, 75)))