import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


model = SentenceTransformer("BAAI/bge-large-en-v1.5")
instruction = "Represent this sentence for searching relevant passages: "


def load_and_prepare_data(file_path):
    """
    Load dataset and split into train/test.
    First 80% -> train_df
    Last 20%  -> test_df
    """
    df = pd.read_csv(file_path)  # must have columns: title, summarized_description, story_point
    df["text"] = "Task title: " + df["title"].fillna('') + "\nTask Description:\n" + df["summarized_description"].fillna('')

    split_index = int(len(df) * 0.8)  # 80% index
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    return train_df, test_df



def create_embeddings(train_df):
    # -------------------
    # 3. Encode training passages (NO instruction)
    # -------------------
    train_passages = train_df["text"].tolist()
    train_embeddings = model.encode(train_passages, normalize_embeddings=True)

    # -------------------
    # 4. Build FAISS index
    # -------------------
    dimension = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity (since we normalized)
    index.add(train_embeddings)
    return index



def retrieve_similar_tasks(index, train_df, test_task, top_k=3):
    query_with_instruction = instruction + test_task
    query_embedding = model.encode([query_with_instruction], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)
    results = train_df.iloc[indices[0]]
    return results, scores[0]
