在孤立词语音识别(Isolated Word Speech Recognition) 中，**DTW**，**GMM** 和 **HMM** 是三种典型的方法：

- **动态时间规整(DTW, Dyanmic Time Warping)** 
- **高斯混合模型(GMM, Gaussian Mixed Model)** 
- **隐马尔可夫模型(HMM, Hidden Markov Model)**

## Introduction
```
  dtw.py: Implementation of Dynamic Time Warping (DTW)
  gmm.py: Implementation of Gaussian Mixture Model (GMM)
  hmm.py: Implementation of Hidden Markov Model (HMM)

  processed_test_records: records with test audios(Chinese)
  processed_train_records: records with train audios(Chinese)
  processed_etest_records: records with test audios(English)
  processed_etrain_records: records with train audios(English)
  utils.py: utils function
  preprocess.py: preprocess audios and split data（由于我已经将数据集分配到了不同的文件夹中，所以我就不提供数据分割代码了）
```

## Launch the script
```
  eg:
  #python preprocess.py (mkdir processed records)
  python dtw.py 
```

## Results
各个方法的数据集和预处理部分完全相同，下面是运行不同文件的结果：
若需要运行中、英文数据集，只需在dtw/gmm/hmm.py文件的主程序中按注释更改路径即可，即(e)
```
python dtw.py
----------Dynamic Time Warping (DTW)----------
Train num: 160, Test num: 40, Predict true num: 31
Accuracy: 0.78
```

```
python gmm.py
---------- Gaussian Mixture Model (GMM) ----------
confusion_matrix: 
 [[4 0 0 0 0 0 0 0 0 0]
 [0 4 0 0 0 0 0 0 0 0]
 [0 0 4 0 0 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0 0]
 [0 0 0 0 4 0 0 0 0 0]
 [0 0 0 0 0 4 0 0 0 0]
 [0 0 0 0 0 0 4 0 0 0]
 [0 0 0 0 0 0 0 4 0 0]
 [0 0 0 0 0 0 0 0 4 0]
 [0 0 0 0 0 0 0 0 0 4]]
Train num: 160, Test num: 40, Predict true num: 38
Accuracy: 0.95
```

```
python hmm.py
---------- HMM (Hidden Markov Model) ----------
confusion_matrix: 
 [[4 0 0 0 0 0 0 0 0 0]
 [0 4 0 0 0 0 0 0 0 0]
 [0 0 3 0 1 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0 0]
 [0 0 0 0 4 0 0 0 0 0]
 [0 0 0 0 1 3 0 0 0 0]
 [0 0 0 0 0 0 3 0 1 0]
 [0 0 0 0 0 0 0 4 0 0]
 [0 0 0 1 0 0 1 0 2 0]
 [0 0 0 0 0 0 0 0 0 4]]
Train num: 160, Test num: 40, Predict true num: 35
Accuracy: 0.875
```