import pandas as pd
import numpy as np
import os
# LangChain and ML Libraries
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def recommend(query: str, category: str, tone: str):
    # 1. Data Loading and Initial Setup
    print("Loading book data...")
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.png",
        books["large_thumbnail"],
    )
    books_dict = books.set_index('isbn13').to_dict('index')

    # 2. Setup Embeddings, VectorDB, and Reranker
    persist_directory = "chroma_db_books_enriched"

    # main architecture is using all-MiniLM-L6-v2
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Initializing reranker model...")
    # use pretrained crossencoder to rerank
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # reload the existing vector db
    if not os.path.exists(persist_directory):
        print("Creating new vector database...")
        documents_to_embed = [
            Document(
                page_content=(
                    f"Title: {book_data.get('title', '')}\n"
                    f"Authors: {str(book_data.get('authors', '')).replace(';', ', ')}\n"
                    f"Categories: {book_data.get('simple_categroies', '')}\n"
                    f"Description: {book_data.get('description', '')}"
                ),
                metadata={"isbn13": int(isbn)}
            ) for isbn, book_data in books_dict.items()
        ]
        db_books = Chroma.from_documents(
            documents_to_embed, hf_embeddings, persist_directory=persist_directory
        )
    else:
        print("Loading existing vector database...")
        db_books = Chroma(
            persist_directory=persist_directory, embedding_function=hf_embeddings
        )

    # 3. Setup Hybrid Search (Semantic + Keyword)
    print("Setting up hybrid search retriever...")
    retrieved_data = db_books.get(include=["metadatas", "documents"])
    all_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(retrieved_data['documents'], retrieved_data['metadatas'])
    ]
    # use bm25 to calculate the relevance
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 50
    # use all-MiniLM-L6-v2 to be second retriever
    vector_retriever = db_books.as_retriever(search_kwargs={"k": 50})
    # ensemble both retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    # 4. The Core Recommendation Logic
    def get_recommendations(query: str, category: str, tone: str):
        retrieved_docs = ensemble_retriever.get_relevant_documents(query)
        # rerank score
        if retrieved_docs:
            cross_encoder_inputs = [[query, doc.page_content] for doc in retrieved_docs]
            scores = reranker.predict(cross_encoder_inputs)
            for i in range(len(scores)):
                retrieved_docs[i].metadata['rerank_score'] = scores[i]
            retrieved_docs = sorted(retrieved_docs, key=lambda x: x.metadata['rerank_score'], reverse=True)

        seen_isbns = set()
        final_isbns = []
        for doc in retrieved_docs:
            isbn = doc.metadata['isbn13']
            if isbn not in seen_isbns:
                seen_isbns.add(isbn)
                final_isbns.append(isbn)

        if not final_isbns:
            return pd.DataFrame()

        recs_df = books[books["isbn13"].isin(final_isbns)].set_index('isbn13').loc[final_isbns].reset_index()

        if category != "All":
            recs_df = recs_df[recs_df["simple_categroies"] == category]

        if tone != "All":
            sort_column = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}.get(tone)
            if sort_column in recs_df.columns:
                recs_df = recs_df.sort_values(by=sort_column, ascending=False)
                
        return recs_df.head(16)
    return get_recommendations(query,category,tone)
