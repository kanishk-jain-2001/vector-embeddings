# imports
from openai import OpenAI  # Imports the OpenAI class from the openai package.
import pandas as pd  # Imports the pandas package and gives it the alias 'pd'.
import tiktoken  # Imports the tiktoken module.
from dotenv import load_dotenv  # Imports the load_dotenv function from the dotenv package.

# Load the OpenAI API Key and get the OpenAI Client Started
load_dotenv()  # Loads environment variables from a .env file into the environment.
client = OpenAI()  # Creates an instance of the OpenAI class.

# Function for getting embeddings
def get_embedding(text, model="text-embedding-ada-002"):  # Defines a function to get embeddings.
   text = text.replace("\n", " ")  # Replaces newline characters in the text with spaces.
   return client.embeddings.create(input=[text], model=model).data[0].embedding  # Returns the embedding of the input text using the specified model.

# embedding model parameters
embedding_model = "text-embedding-ada-002"  # Specifies the model to use for embeddings.
embedding_encoding = "cl100k_base"  # Specifies the encoding for the text-embedding-ada-002 model.
max_tokens = 8000  # Sets the maximum number of tokens for the embedding model.

# load & inspect dataset
input_datapath = "Reviews.csv"  # Path to the input dataset file.
df = pd.read_csv(input_datapath, index_col=0)  # Reads the dataset into a pandas DataFrame.
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]  # Selects specific columns from the DataFrame.
df = df.dropna()  # Removes rows with missing values.
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)  # Combines 'Summary' and 'Text' columns into a new 'combined' column.
df.head(2)  # Displays the first two rows of the DataFrame.

# subsample to 1k most recent reviews and remove samples that are too long
top_n = 1000  # Sets the number of most recent reviews to keep.
df = df.sort_values("Time").tail(top_n * 2)  # Sorts the DataFrame by 'Time' and keeps the last 2000 entries.
df.drop("Time", axis=1, inplace=True)  # Removes the 'Time' column from the DataFrame.

encoding = tiktoken.get_encoding(embedding_encoding)  # Gets the encoding for the embedding model.

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))  # Adds a new column 'n_tokens' representing the number of tokens in each review.
df = df[df.n_tokens <= max_tokens].tail(top_n)  # Keeps only the reviews with a token count within the limit.
len(df)  # Outputs the length of the DataFrame.

# Get the embeddings This may take a few minutes
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model=embedding_model))  # Adds a new column 'embedding' containing the embeddings for each review.
df.to_csv("fine_food_reviews_with_embeddings_1k.csv")  # Saves the DataFrame to a CSV file.
