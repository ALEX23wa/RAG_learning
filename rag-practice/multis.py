import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

MODEL_NAME = "Visualized_m3"
MODEL_PATH = "RAG_learing/Models/Visualized_m3.pth"
DATA_DIR = "RAG_learing/rag-practice/papers"
MILVUS_URI = "http://localhost:19530"

class Encoder:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Visualized_BGE(model_name_bge=model_name,model_weight=model_path)
        self.model.to(self.device)
        self.model.eval()

    def encode_query(self, text, image_path):
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]


    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]

    
