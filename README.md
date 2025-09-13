# 📚 LLM Book Recommender

An **AI-powered book recommendation system** that helps users discover their next favorite read by understanding natural language input.  
Simply type in a phrase like *"a story about finding hope in adversity"* and get highly relevant, personalized book suggestions.  

Built with **Gradio** for an interactive, user-friendly web experience and powered by modern **LLM + IR techniques**.

---

## ✨ Features
- 🔍 **Natural Language Search** – Express your mood, themes, or ideas in plain English.  
- 📖 **7k Books Dataset** – Recommendations are based on a rich collection of books with detailed metadata.  
- 🎨 **Interactive Gradio UI** – Clean, intuitive, and visually appealing interface for book discovery.  
- 🤖 **LLM-Powered Pipeline** – From classification to embeddings, retrieval, and reranking.

---

## 🖼️ Interface Preview
<div align="center">
  
[![UI Screenshot 1](https://s21.ax1x.com/2025/09/13/pVWR7h6.md.png)](https://imgse.com/i/pVWR7h6)  
[![UI Screenshot 2](https://s21.ax1x.com/2025/09/13/pVWRb9K.md.png)](https://imgse.com/i/pVWRb9K)  

</div>

---

## 📊 Dataset
- **Source:** [7k Books with Metadata – Kaggle](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)  
- Contains **7,000+ books** with metadata (title, author, genre, description, etc.)  

---

## 🧠 Model Pipeline
### 🔧 Preprocessing
- **Zero-Shot Classification:** `facebook/bart-large-mnli`  
- **Emotion & Sentiment Analysis:** `j-hartmann/emotion-english-distilroberta-base`  

### 🔎 Embeddings & Retrieval
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Retriever:** `BM25Retriever`  

### 🎯 Reranking
- **Cross-Encoder Reranker:** `ms-marco-MiniLM-L-6-v2`  

---

## 🚀 How It Works
1. **User Input:** e.g. "an adventurous story set in space"  
2. **Preprocessing:** Extract emotions, themes, and intent  
3. **Embedding & Retrieval:** Find candidate books using semantic search + BM25  
4. **Reranking:** Refine results with a cross-encoder for maximum relevance  
5. **Display Results:** Beautiful book cards shown in the Gradio app  

---

## 🛠️ Tech Stack
- **Frontend:** Gradio  
- **Backend:** Python (Hugging Face Transformers, Sentence-Transformers)  
- **IR Tools:** BM25, Semantic Search  
- **Dataset:** 7k Books Metadata  

---

## 📌 Future Improvements
- 📚 Add user profile + reading history for personalized recommendations  
- 🌍 Support for multiple languages  
- 🔄 Real-time fine-tuning with feedback  

---

## 💡 Inspiration
This project was inspired by the idea of **discovering books through meaning rather than keywords**, making book recommendations feel like a conversation with a librarian who truly understands you.

---
