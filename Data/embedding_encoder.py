import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import os


################################################
################## ENCODER  ####################
################################################

class EmbeddingEncoder:
    def __init__(self, model_name=None, name_column_emb='sbert_embeddings', type_emd='glove', device='cuda'):
            
        if type_emd == 'bert':
            self.model = BERTEmbeddingEncoder(model_name, device) 

        elif type_emd == 'sentencebert':
            self.model = SetenceBERTEmbeddingEncoder(model_name, name_column_emb, device) 
      
        else : raise Exception ("'model' & 'model_name' are None type, at least one is requered")
        
        
    def forward(self, dataset, text_column='text'):
        
        return self.model.forward(dataset, text_column)
    

################################################
################## ABSTRACT ####################
################################################

class BaseEmbeddingEncoder(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name

    @abstractmethod
    def forward(self, dataset):
        pass
    

################################################
############## SentenceBERT  ###################
################################################  


class SetenceBERTEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, model_name, name_column_emb, device):
        super().__init__(model_name)

        self.model_name = model_name
        self.device = device
        self.setencebert_model = SentenceTransformer(model_name, device=device)
        self.name_column_emb = name_column_emb 


    def forward(self, dataset, text_column="text"):

        def compute_embedding(batch):
                texts = batch[text_column]
                emb = self.setencebert_model.encode(
                    texts,
                    convert_to_tensor=True,
                    batch_size=32
                ).cpu().numpy()

                return {self.name_column_emb: np.array(emb)}

        dataset = dataset.map(
            compute_embedding,
            batched=True,
            batch_size=64
        )
        
        return dataset

################################################
#################### BERT  #####################
################################################   

class BERTEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, model_name, device):
        super().__init__(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = device
        self.model.to(device)
        self.model.eval()

    def forward(self, dataset, text_column="text"):
        
        def compute_embeddings(batch):
            texts = batch[text_column]

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state        
            cls_emb = last_hidden[:, 0, :]              
            mean_emb = last_hidden.mean(dim=1)            

            return {
                "bert_cls": cls_emb.cpu().numpy(),
                # "bert_embedding": last_hidden.cpu().numpy(),
                "bert_embedding_mean": mean_emb.cpu().numpy(),
            }

        dataset = dataset.map(
            compute_embeddings,
            batched=True,
            batch_size=32, 
            keep_in_memory=True
        )

        return dataset