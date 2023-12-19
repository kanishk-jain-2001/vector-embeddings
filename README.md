# vector-embeddings
Learning about vector embeddings from OpenAI Docs

# What are embeddings? 

OpenAI's text embeddings measure the relatedness of text strings. 

They are commonly used for:
- Search
- Clustering
- Recommendations
- Anomaly detection 
- Diversity Measurement 
- Classificiation 

Specifically, an embedding is a vector of floating point numbers. The distance between two vectors measures their relatedness. 

# Vector Databases 

Vector Databases are great for knowledge retrieval applications and will help reduce halluncinations by providing the LLM with the relevant context to answer questions. 

# Example 

Let's go along with the example that OpenAI gives on their website, for Amazon fine-food reviews. 
I have downloaded and put the 'Reviews.csv' file we will be working with in the folder for this project, but will be ignored by git as the file is too large. You can download it here: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

There are various python files that each accomplish a different task related to embeddings. For example, getembeddings.py actually acquires the embeddings from the Amazon fine-food reviews page. 

