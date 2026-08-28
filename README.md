# 📚 Transformer-Based Manga Recommendation System

An AI-powered manga recommendation system built using **Transformer embeddings + FAISS vector search**, with real-time metadata enrichment via AniList API.

---

## 🚀 Overview

This project recommends manga based on **semantic similarity** instead of simple keyword matching. It uses transformer-based embeddings to understand the meaning of descriptions and provides highly relevant recommendations.

The system is designed with a **production-ready mindset**, including modular architecture, API integration, error handling, and scalable deployment capability.

---

## 🧠 Key Features

* 🔍 Semantic search using transformer embeddings
* ⚡ Fast similarity search using FAISS
* 🌐 Real-time metadata via AniList GraphQL API
* 🛡️ Robust error handling with retries (Tenacity)
* 🖼️ Safe image loading with fallback support
* 🎛️ Interactive UI with Streamlit
* 🧩 Modular code structure (DataLoader, ModelLoader, Recommender, APIHandler)

---

## 🏗️ System Architecture

Frontend (Streamlit)
↓
Backend (FastAPI - planned / extendable)
↓
FAISS Vector Index + Transformer Model
↓
External API (AniList)

---

## 🛠️ Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* NumPy, Pandas
* Requests
* Tenacity (retry handling)
* PIL (image handling)

---

## 📂 Project Structure

```
├── app.py / app2.py        # Main Streamlit application
├── semantic_manga_dataset.csv
├── manga_embeddings.npy
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app2.py
```

---

## 📊 How It Works

1. Load dataset and precomputed embeddings
2. Normalize embeddings and store in FAISS index
3. Convert user-selected manga into semantic query
4. Perform similarity search using FAISS
5. Re-rank results using:

   * Genre overlap
   * Title similarity
   * Popularity
6. Fetch additional metadata from AniList API
7. Display results with images and descriptions

---

## 🔥 Future Improvements

* [ ] Convert to FastAPI backend
* [ ] Deploy on cloud (AWS EC2 + S3)
* [ ] Dockerize the application
* [ ] Add user-based recommendation system
* [ ] Add authentication & logging
* [ ] Optimize inference latency

---

## ☁️ Deployment Plan

* Backend API using FastAPI
* Model hosting on AWS EC2
* Storage via AWS S3
* Public access via Cloudflare Tunnel

---

## 💡 Learning Outcomes

* Built end-to-end AI recommendation system
* Implemented vector similarity search
* Integrated external APIs in ML workflow
* Applied production-level error handling
* Designed scalable AI architecture

---

## 👨‍💻 Author

**Nikhil Saroj**
📧 [nikhilsaroj.ai@gmail.com](mailto:nikhilsaroj.ai@gmail.com)
🔗 GitHub: https://github.com/Nikhildsaroj

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

---
