# imports
import pandas as pd
from openai import OpenAI  
import numpy as np
from ast import literal_eval
from dotenv import load_dotenv

# Load the OpenAI API Key and get the OpenAI Client Started
load_dotenv()  # Loads environment variables from a .env file into the environment.
client = OpenAI()  # Creates an instance of the OpenAI class.

datafile_path = "fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)


# Functions 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):  # Defines a function to get embeddings.
   text = text.replace("\n", " ")  # Replaces newline characters in the text with spaces.
   return client.embeddings.create(input=[text], model=model).data[0].embedding  # Returns the embedding of the input text using the specified model.


# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


results = search_reviews(df, "delicious beans", n=3)
