# Code_aaai：

**Updates Dec 14 2020** ： This is our first version , we'll keep updating the repository.  

**Code for AAAI 2021 paper [Exploring Explainable Selection to Control Abstractive Summarization]**

**In this repository, we can provide you the following things:**

The two sub-directorys conclude the BERT version and Transforer version code of ESCA.
* The required environment of the code.
* How to build dataset for model train. 
* How to Train the model & hybrid model(extractor and abstractor).
* How to build dataset for model test. 
* How to Test the model.

## Requirements
* Python 3.6
* torch 1.1.0
* Pytorch-transformers 1.2.0
* Pyrouge 0.1.3 (for evaluation)
* Standford CoreNLP 3.8.0 (for data preprocessing)

**Note**: If you need to use the lstm-rnn as the encoder, you need to establish the dataset follow [this repository](https://github.com/abisee/cnn-dailymail), using the Standford CoreNLP and NLTK um...

**Note**: To use ROUGE evaluation, you need to download the 'ROUGE-1.5.5' package and then use pyrouge.

**Error Handling**: If you encounter the error message `Cannot open exception db file for reading: /path/to/ROUGE-1.5.5/data/WordNet-2.0.exc.db` when using pyrouge, the problem can be solved from [here](https://github.com/tagucci/pythonrouge#error-handling) and you need to be patient because the ROUGE is very queer.

## Train dataset
### 1.Download the precessed dataset

Precessed dataset will be pulled on there later, you can unzip the zipfile and put `.pt` files into `bert_data`.

### 2.Process the dataset by self (CNN/DailyMail)

#### 1> Download Stories
Download and unzip the `stories` directories from [DMQA](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

#### 2> Download Stanford CoreNLP and export
Stanford CoreNLP is a Word segmentation tools, Dowload it from [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
* Replacing `/path/to/` with the path to where you saved the standford-corenlp-full-2o17-o6-o9 directory.

#### 3> Sentence Splitting and Tokenization
```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```
* `Raw_path` is the directory containing story files.
* `Json_path` is the target directory to save generated json files.

#### 4> Format to Simpler Json Files
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```
* `Map_path` is the directory containing the urls files, which we have provided.

#### 5> Format to pt Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```
* `Bert_data_path` is the target directory to save the generated binary files (bert_data).

## How to Train the hybrid model (extractor and abstractor)?

### 1. Pretrain the extractor
```
python train.py --pairwise -task ext -mode train -data_path DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus x -report_every 100 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 100000 -accum_count 2 -log_file LOG_PATH -use_interval true -warmup_steps 10000 -max_pos 512
```

* We use the cross entropy loss to train the extractor, expecting better parameters. Then we proposed a new loss `pairwise loss` to learn the relationship between sentences.

### 2. Pretrain the abstractor
```
python train.py -task abs -mode train -data_path DATA_PATH -dec_dropout 0.2  -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus x  -log_file LOG_PATH
```
* We reference the Pointer-Generator network structure, and use the P_gen calculates the vocab_prob based on ... And you can choose not to pretrain the abstractor because of it'll be trained in Hybrid mode, too.

### 3. Train the Hybrid
```
python train.py -train_from_extractor EXTRACTOR_CHECKPOINT_PATH -train_from_abstractor ABSTRACTOR_CHECKPOINT_PATH --hybrid_loss -lr 0.002 -task hybrid -mode train -model_path=MODEL_PATH -data_path DATA_PATH -ext_dropout 0.1 -visible_gpus x -report_every 50 -save_checkpoint_steps 1800 -batch_size 2000 -train_steps 200000 -accum_count 5  -log_file LOG_PATH -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 
```
* there we offer three extra parameters methods: `--oracle/--hybrid_connector/--hybrid_loss/` to train hybrid model, `Overlap, NoT Juxtaposition`.

#### The transformer version is like the above we provide...

## How to val and test the MODEL by ROUGE or builded dataset?

### 1.Test by ROUGE：
```
python train.py -test_from MODEL_PATH -task hybrid -mode test -batch_size 3000 -test_batch_size 500 -data_path DATA_PATH -log_file LOG_PATH -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus x -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path RESULT_PATH
```
### 2.Validate by ROUGE:
```
python train.py -task hybrid -mode validate -batch_size 15000 --hybrid_loss -control Rel -test_batch_size 15000 -data_path DATA_PATH -log_file LOG_PATH -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path RESULT_PATH -test_all 
```
* You can choose the `-test_all` parameter, the system'll load all checkpoints and select the top ones to generate summaries, IF you have enough time.

### 3.Build dataset for test:

#### RELVANCE:
We searched a lot of literature and find the CNN/DailyMail dataset have a version with title. but need some binary conversion is required to restore the original version, then we add the title information into reference summary as the dataset to test the relavance of system.

#### NOVELTY:
Inorder to test the diversity of System, we use the Advanced unsupervised extraction system `PacSum`. We delete the first five sentences from the original text as INPUT and enter it into Pacsums and we take the output as reference to test the performance of Model. 














