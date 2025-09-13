## LLM Book Recommender
### About the Project
This AI-powered book recommendation application uses the "7k books" dataset from a Kaggle competition as its database. It aims to provide highly relevant and personalized book recommendations by understanding natural language user input (e.g., "a story about finding hope in adversity"). The project features a beautiful, interactive web user interface built with Gradio, allowing users to easily discover their next favorite book.
### Interface Example
[![pVWR7h6.md.png](https://s21.ax1x.com/2025/09/13/pVWR7h6.md.png)](https://imgse.com/i/pVWR7h6)
[![pVWRb9K.md.png](https://s21.ax1x.com/2025/09/13/pVWRb9K.md.png)](https://imgse.com/i/pVWRb9K)
### Data source
·**7k books**
[Kaggle link](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

### Model Used
#### Preprocessing model
·`zero-shot-classification model`:"facebook/bart-large-mnli"
·`sentiment-analysis`: "j-hartmann/emotion-english-distilroberta-base"
#### Embeddings and Key search
·`HuggingfaceEmbeddings`: "sentence-transformers/all-MiniLM-L6-v2"
·`Key retreiver`: "BM25Retriever"
#### Reranker
·`ms-marco-MiniLM-L-6-v2`