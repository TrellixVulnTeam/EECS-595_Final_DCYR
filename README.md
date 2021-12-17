# Lex2Vec: A Modified Language Model Combining WSD and Lexical Information with BERT

<p align="center">
  <img align="center" src="https://github.com/You-Ivan/EECS-595_Final/blob/main/modle%20structure.png" alt="...">
</p>

## Dependencies
The code was written with, or depends on:
* Python 3.6
* Pytorch 1.4.0
* NLTK 3.4.5
* [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2]

## Running the code
1. Create a virtualenv and install dependecies
      ```bash
      virtualenv -p python3.6 env
      source env/bin/activate
      pip install -r requirements.txt
      python -m nltk.downloader wordnet
      python -m spacy download en
      ```         
1. Fetch data and pre-process. This will create pre-processed files in data folder. (In case there is an issue handling large files, processed input word embeddings ```i_id_embedding_glove.p``` are also provided)
      ```bash
      bash fetch_data.sh  
      bash preprocess.sh data
      ```     
1.  * To train ConvE embeddings, change directory to the ```conve``` folder and refer to the [README](./conve/README.md) in that folder. Generate embeddings for the WSD task:
      ```bash
      python generate_output_embeddings.py ./conve/saved_embeddings/embeddings.npz data conve_embeddings  
      ```    
    * Alternatively, to use pre-trained embeddings, copy the pre-trained conve embeddings (```o_id_embedding_conve_embeddings.npz```) to the ```data``` folder.
1.  Train a WSD model. This saves the model with best dev set score at ```./saved_models/model.pt```.
      ```bash
      CUDA_VISIBLE_DEVICES=0 python wsd_main.py --cuda --dropout 0.5 --epochs 200 --input_directory ./data --scorer ./ --output_embedding customnpz-o_id_embedding_conve_embeddings.npz --train semcor --val semeval2007 --lr 0.0001 --predict_on_unseen --save ./saved_models/model.pt
      ```
1. Test a WSD model (the model is assumed to saved at ```./saved_models/model.pt```.
      ```bash
      CUDA_VISIBLE_DEVICES=0 python wsd_main.py --cuda --dropout 0.5 --epochs 0 --input_directory ./data --scorer ./ --output_embedding customnpz-o_id_embedding_conve_embeddings.npz --train semcor --val semeval2007 --lr 0.0001 --predict_on_unseen --evaluate --pretrained ./saved_models/model.pt
      ```
      
## Pre-trained embeddings and models
All files are shared at https://drive.google.com/drive/folders/1zxTtepYpolF3Qwj3BtaNLBi0ypusamAt?usp=sharing
Uncompress model files using gunzip before using.
A & B would suffice if only training/evaluating a WSD model.

A. Pre-trained conve embeddings: ```o_id_embedding_conve_embeddings.npz```

B. Pre-trained model: ```model.pt.gz``` (F1 score on ALL dataset: 72.1)

C. Pre-trained ConvE model: ```WN18RR_conve_0.2_0.3__defn.model.gz```

D. Processed input word embeddings: ```i_id_embedding_glove.p``` (Needed only if there are issues handling large files during preprocessing)


An earlier version contained some code for weighted cross entropy loss (now enabled only by the ```--weighted_loss``` flag). The scheme wasn't really helpful and is not recommended. However, a pre-trained model for the same is shared: ```model_weighted.pt.gz``` (F1 score on ALL dataset: 72.1)
 
