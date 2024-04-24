


# Text classification with FinnSentiment

For running a BERT based model for a text classification (e.g. sentiment analysis) task. In order to run the notebooks, you would need enough Finnish text data at hand (at least thousands of samples, preferably more). The dataset used here is openly available [FinnSentiment](https://korp.csc.fi/download/finsen/src/). 

This repository contains the following notebooks that should be run in order:
1. preprocessing
2. pretraining (masked language modeling)
3. finetuning (model training)

The pretraining is unsupervised, so labels are not needed for it. For the finetuning there should be e.g. manually added labels for all of the samples. In BERT modeling, large-scale pretrained models are typically utilized to specialize your own finetuned model to downstream task(s). Before finetuning, you can add (further) pretraining (with MLM) for enhancing model performance. The backbone, in the notebooks, is currently set to FinBERT-base (Virtanen et al., 2019) as this repository has been built to deal with Finnish text data.

With other datasets, for running the pretraining notebook, each dataset should be in a csv, json or txt file with the data being in a column named "text" or on the first column. For running the finetuning notebook, each dataset should be in a csv file with the data being in a column named "text" and the information of labels in column named "label".


# Installation

Use the following commands to install libraries needed to run the notebooks:
```
git clone https://github.com/DARIAH-FI-Survey-Concept-Network/finnsentiment-classification.git
cd finnsentiment-classification
pip install -r requirements.txt
```


## Code tested with
- Python 3.7
- Torch 1.7.1
- Transformers 4.16.2
- Pandas 1.3.5
- Scikit-learn 1.0.2


## Things to note
- This repository was released as part of FIN-CLARIAH infrastructure project (2022-2023). 

## Dataset
Lindén, K., Jauhiainen, T., & Hardwick, S. (2020). FinnSentiment--A Finnish Social Media Corpus for Sentiment Polarity Annotation. arXiv preprint arXiv:2012.02613. https://arxiv.org/pdf/2012.02613.pdf


## References
- https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
- https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-itpt
- https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently


