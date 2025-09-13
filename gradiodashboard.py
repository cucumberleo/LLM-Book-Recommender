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

# Gradio
import gradio as gr

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

# 5. Gradio UI Functions
def process_recommendations(query, category, tone):
    if not query:
        gr.Info("Please enter a description to find book recommendations.")
        return [], None, gr.Markdown(visible=False), {}
        
    recommended_books_df = get_recommendations(query, category, tone)

    if recommended_books_df.empty:
        gr.Warning("No books found matching your criteria. Try a broader search!")
        return [], None, gr.Markdown(visible=False), {}
    
    # CHANGE 1: Prepare the dictionary to be stored in the state
    recommendations_dict = recommended_books_df.set_index('isbn13').to_dict('index')
    gallery_output = [(row["large_thumbnail"], row["title"]) for _, row in recommended_books_df.iterrows()]
    first_book_isbn = recommended_books_df.iloc[0]['isbn13']
    # Pass the dict directly to the details function
    details_output = get_book_details_markdown(first_book_isbn, recommendations_dict)
    
    # CHANGE 2: Return the dictionary to be saved in gr.State
    return gallery_output, details_output, gr.Markdown(visible=True), recommendations_dict

# CHANGE 3: Function now accepts the recommendations dictionary from the state
def get_book_details_markdown(selected_isbn, recommendations_dict):
    if not selected_isbn or not recommendations_dict:
        return ""
        
    book = recommendations_dict.get(int(selected_isbn))
    if not book:
        return "Book details not found."

    authors_list = str(book.get("authors", "N/A")).split(";")
    if len(authors_list) == 1:
        authors_str = authors_list[0]
    elif len(authors_list) <= 2:
        authors_str = " and ".join(authors_list)
    else:
        authors_str = f"{', '.join(authors_list[:-1])}, and {authors_list[-1]}"

    description = str(book.get("description", "No description available."))
    desc_snippet = " ".join(description.split()[:70]) + "..." if len(description.split()) > 70 else description

    md = f"""
    ### {book.get('title', 'No Title')}
    **✍️ By:** {authors_str}
    **📚 Category:** {book.get('simple_categroies', 'N/A')}
    **⭐ Average Rating:** {book.get('average_rating', 'N/A')} / 5
    ---
    **📖 Description:**
    {desc_snippet}
    """
    return md

# CHANGE 4: Function now accepts the recommendations_state as an input
def on_gallery_select(evt: gr.SelectData, recommendations_dict: dict):
    selected_title = evt.value
    for isbn, book_data in recommendations_dict.items():
        if book_data['title'] == selected_title:
            # Pass the dict to the details function
            return get_book_details_markdown(isbn, recommendations_dict)
    return "Could not find details for the selected book."

# --- 6. Gradio Interface ---
categories = ["All"] + sorted(books["simple_categroies"].dropna().unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css="footer {display: none !important}") as dashboard:
    gr.Markdown("# 📚 Advanced Book Recommender")
    gr.Markdown("Describe the kind of book you want to read. Our powerful engine will find the perfect match for you!")
    
    # CHANGE 5: Add the gr.State component to manage session data
    recommendation_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(label="What are you in the mood for?", placeholder="e.g., A thrilling sci-fi adventure...")
            with gr.Row():
                category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
                tone_dropdown = gr.Dropdown(choices=tones, label="Emotional Tone", value="All")
            submit_button = gr.Button("Find My Book ✨", variant="primary")
            gr.Examples(
                examples=[
                    "A story about overcoming loss and finding hope.",
                    "A funny and lighthearted romance novel.",
                    "A dark fantasy world with complex magic systems.",
                ],
                inputs=user_query,
                label="Example Searches"
            )
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### Recommendations")
                output_gallery = gr.Gallery(label="Recommended Books", columns=4, rows=2, object_fit="contain", height=450, show_label=False)
            with gr.Group():
                gr.Markdown("### Book Details")
                output_details = gr.Markdown("Click on a book cover above to see details here.", visible=False)

    # Event Handlers
    # Update the button's click event to output to the state
    submit_button.click(
        fn=process_recommendations,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[output_gallery, output_details, output_details, recommendation_state]
    )
    output_gallery.select(
        fn=on_gallery_select,
        inputs=[recommendation_state],
        outputs=output_details
    )

if __name__ == "__main__":
    print("Launching Gradio Dashboard...")
    dashboard.launch(share=True)