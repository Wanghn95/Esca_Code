# Code_aaai：

**Updates Dec 14 2020** ： This is our first version , we'll keep updating the repository.  

**Code for AAAI 2021 paper [Exploring Explainable Selection to Control Abstract Generation]**

**In this repository, we can provide you the following things:**

The two sub-directorys conclude the BERT version and Transforer version code of ESCA.
* The required environment of the code.
* How to build dataset for model train correlation. 
* How to Train the model & hybrid model(extractor and abstractor).
* How to build dataset for model test correlation. 
* How to Test the model.
* The hyper-parameters you need to pay attention to.

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
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```
* `Bert_data_path` is the target directory to save the generated binary files (bert_data).





