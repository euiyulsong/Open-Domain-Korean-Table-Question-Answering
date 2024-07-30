# Open Domain Question Answering
RAFT, TAPT, Instruction Tuning, and Preference Optimization on [google/gemma-2b](https://huggingface.co/google/gemma-2b) for Open Domain Question Answering
## Performance
### Hyperparameters
| Hp | Retrieval | Reranker | Reader | 
| ------------- | ------------- | ------------- | ------------- |
| lr |  |  |  |
| bs |  |  |  |
| scheduler |  |  |  |
| gpu | rtx 4090  | rtx 4090  | rtx 4090  |
| τ | 0.02 |  |  |
### Retrieval
| Method | R-Precision | Recall@5 | TPS | 
| ------------- | ------------- | ------------- | ------------- |
| BM25 | 67.169 | 91.265 | 0.007s |
| Dense | 74.398  | 94.127 | 0.44s |
| Cross Encoder | 86.898 | 95.783 | 1.30s |

### Reader
| Method | EM | F-1 | Rouge-L |
| ------------- | ------------- | ------------- |------------- |
| SFT |  |   | |
| RAFT |   |   | |
| Inst + RAFT |   |   | |
| Inst + TAPT + RAFT |   |   | |
| Inst + TAPT + Synthetic RAFT + RAFT |   |   | |
| Inst + TAPT + Synthetic RAFT + SimPO|   |   | |
| Inst + TPT + Synthetic SimPO + SimPO |   |   | |

## Directory Structure
```
├── README.md
├── refine.py #perform data cleaning
└── requirements.txt
```

## Install Required Packages
```pip install -r requirements.txt```

## Data Preprocessing
### Clean Data
```
python src/clean/clean.py # html parsing
python src/clean/map.py # map negative pairs and filter data with more than 1 gold table for Retrieval
python src/clean/negatives.py # clean corpus and map rejection samples for Preference Optmization
```
### Train/Test split
```
bash src/split/split.sh
```

### Synthetic Data Generation
```
python src/synthetic/question_generation.py # question generation
python src/synthetic/answer_generation.py # qnswer generation
```

## Retrieval
### Data Preparation
```python src/clean/copy_train.py```
### Training
#### Dense Retrieval
```bash src/retrieval/dense_finetuning.sh```
#### Cross Encoder Reranker
```bash src/retrieval/ce_finetuning.sh```
### Inferenece
```python src/retrieval/retrieval.py```



## SFT

## Instruction Tuning

## Simple Preference Optimization
