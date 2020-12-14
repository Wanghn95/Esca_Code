# Code_aaai：
**Updates Dec 14 2020** ： This is our first version , we'll keep updating the repository.  

**Code for AAAI 2021 paper [Exploring Explainable Selection to Control Abstract Generation]**

**In this repository, we can provide you the following things:**

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
* Standford CoreNLP 3.7.0 (for data preprocessing)

**Note**: If you need to use the lstm-rnn as the encoder, you need to establish the dataset follow [this repository](https://github.com/abisee/cnn-dailymail), using the Standford CoreNLP and NLTK um...

**Note**: To use ROUGE evaluation, you need to download the 'ROUGE-1.5.5' package and then use pyrouge.

**Error Handling**: If you encounter the error message `Cannot open exception db file for reading: /path/to/ROUGE-1.5.5/data/WordNet-2.0.exc.db` when using pyrouge, the problem can be solved from [here](https://github.com/tagucci/pythonrouge#error-handling) and you need to be patient because the ROUGE is very queer.

## 
