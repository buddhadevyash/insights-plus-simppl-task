# Insights Plus: Social Media Analyst

This AI-driven web application provides a comprehensive solution for **analyzing and visualizing social media data** from YouTube and Reddit. It transforms raw data into actionable intelligence, empowering users to uncover insights, monitor engagement, and interact with data using natural language queries.  

The project leverages a modern and scalable architecture with:  
- **Next.js, Javascript, ShadCN, Vis.js** for a responsive frontend.  
- **FastAPI** with Python for a high-performance backend.  
- **Neo4j** for graph-based data modeling.  
- **Groq API** for cutting-edge AI-powered analysis.  

[![Next.js](https://img.shields.io/badge/Next.js-15.0.0-black.svg)](https://nextjs.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![ShadCN/UI](https://img.shields.io/badge/ShadCN/UI-Components-purple.svg)](https://ui.shadcn.com/)
[![Vis.js](https://img.shields.io/badge/Vis.js-Network%20Graphs-orange.svg)](https://visjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-008CC1.svg)](https://neo4j.com/)
[![Groq API](https://img.shields.io/badge/Groq-API-red.svg)](https://groq.com/)


<img width="962" height="743" alt="- visual selection" src="https://github.com/user-attachments/assets/4ad45035-80a4-49d1-9dae-c9413b8e2fa0" />


---
    ## 📽️ Demo Video  
[![Watch the Demo: https://www.youtube.com/watch?v=yD7N7sRS68E](https://img.youtube.com/vi/yD7N7sRS68E/0.jpg)](https://youtu.be/yD7N7sRS68E)
# Technical System Overview

This project integrates **conversational AI, clustering, interactive dashboards, graph-based data modeling, and external data augmentation** to deliver advanced insights from YouTube and Reddit datasets. Below is a detailed breakdown of the components I designed.
<img width="1014" height="746" alt="- visual selection(1)" src="https://github.com/user-attachments/assets/8845452b-340f-4bb0-9333-fbaa247cb11a" />



---

### 1. Conversational AI–Driven Smart Reporting  
I designed a **conversational analytics engine** that transforms raw social media data into **inferential visualizations and reports** based on natural language queries.  

- **Embedding Generation:** Used `all-MiniLM-L6-v2` to convert unstructured text (posts, comments, discussions) into semantic embeddings.  
- **Similarity Search:** Leveraged **cosine similarity** with **KNN retrieval** using embeddings from **`all-MiniLM-L6-v2`** to extract the most contextually relevant posts.
**Context Optimization:** Limits the prompt to top relevant records to respect Groq’s token limits, dynamically pruning less relevant content.
- **Reporting Framework:**  
  - **Summary Reports** → key engagement metrics, sentiment distributions, and posting frequencies.  
  - **Detailed Drilldowns** → fine-grained insights at the post/comment level.  
  - **Key Metrics Extraction** → automatically highlights top-performing entities.  
- **Inference Engine:** Integrated **OpenAI OSS 20B** on Groq, balancing performance with low-latency response times.  

This design ensures scalability with larger context windows when upgrading beyond free-tier Groq limits.  
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-10" src="https://github.com/user-attachments/assets/39ff6b1d-06b1-4bb2-be96-9f49be4423d4" />

<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-21" src="https://github.com/user-attachments/assets/cbdd440b-a14f-438e-b7f2-40ad254c6c03" />
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-28" src="https://github.com/user-attachments/assets/9ac901c2-ffc9-4071-bb03-08d7eddb1820" />
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-33" src="https://github.com/user-attachments/assets/4df9422b-8401-48b8-83bd-71432c1ff255" />
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-36" src="https://github.com/user-attachments/assets/b7802a68-7039-4f3d-8ebe-4f330ec34654" />
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-40" src="https://github.com/user-attachments/assets/dc2dfec0-db83-41fe-a543-4b01270b4781" />

<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-42" src="https://github.com/user-attachments/assets/d5a6d7dc-f59d-4a81-a02c-f9585b2f5bed" />
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-46" src="https://github.com/user-attachments/assets/94a98424-3f91-4e7b-89bb-d77e4712f1d1" />
<img width="1748" height="1013" alt="Screenshot From 2025-09-11 01-30-49" src="https://github.com/user-attachments/assets/f0bc7236-28f8-4999-a872-f3090de253f4" />

---

### 2. Cluster Querying & Tag-Based Exploration  
I developed a **semantic clustering pipeline** to automatically group related content into coherent themes, enabling both visualization and direct query access.  

- **Clustering:** Applied **KMeans** on embeddings to form **top-k thematic clusters**, capturing high-level discussion topics, mapped dynamically by Vis.js on the frontend.  
- **Cluster Querying:** Implemented a `@ClusterName` tagging system, inspired by WhatsApp mentions, for easy referencing of specific clusters.  
- **Visualization:** Generated **cluster visualizations** to observe data point distributions, intra-cluster cohesion, and inter-cluster separation.  
- **Cluster Labeling:** Leveraged **LLaMA 8.1-Instant** to automatically assign descriptive, human-readable names to clusters (e.g., *AI Research Trends*, *Gaming Hardware*, *Community Feedback*).  


This enables me to move seamlessly between **macro-level trend discovery** and **micro-level cluster drilldowns**.  
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-29-52" src="https://github.com/user-attachments/assets/9da42f14-1af9-49c1-996f-d41aeec1dac2" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-29-56" src="https://github.com/user-attachments/assets/3b01c35a-7d8c-4ca0-9b54-5a861ef0b01c" />

<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-30-01" src="https://github.com/user-attachments/assets/e9d1d088-bb53-42b9-9e9c-ad8f13088f70" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-31-18" src="https://github.com/user-attachments/assets/e77a50e7-fea2-4194-875c-1b7bf3751105" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-30-47" src="https://github.com/user-attachments/assets/23d65030-2a93-4bd6-9325-401fb7b78cbb" />



---

### 3. Dynamic Real-Time Dashboard  
I implemented a **real-time interactive dashboard** to centralize insights and provide at-a-glance intelligence.  

- **Data Refreshing:** Designed for **dynamic updates** as new social data is ingested.  
- **Smart Filters:** Added filters for platform (YouTube/Reddit), sentiment range, engagement thresholds, and time periods.  
- **Leaderboards:** Automatically generates rankings of **top posts, authors, and clusters** to surface key contributors and discussions.  
- **Visualization Suite:** Includes engagement timelines, sentiment dynamics, keyword trends, and posting velocity histograms.  

This dashboard acts as a **command center**, making complex analysis accessible in a single interface.  

<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-37-10" src="https://github.com/user-attachments/assets/55308df8-e341-4477-8437-fea454f70d58" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-37-13" src="https://github.com/user-attachments/assets/aedc77ea-fa14-4148-af23-49a305c8f91e" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-37-20" src="https://github.com/user-attachments/assets/eced9a83-67a2-43dd-9e25-6dd23dc92292" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-37-36" src="https://github.com/user-attachments/assets/0cce88ed-dda5-463d-9f8b-44164a2786b8" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-37-57" src="https://github.com/user-attachments/assets/c74da94a-ece0-4a34-a808-2b855b5419cf" />



---

### 4. News API Integration (Future Scope)  
As a future extension, I scoped out **News API integration** to contextualize platform-specific insights with **external events and real-world signals**.  

- **Embedding Augmentation:** External articles will be embedded alongside social data to enrich cluster context.  
- **Cross-Domain Correlation:** Enables tracing correlations between trending news and social platform engagement spikes.  
- **Optimal Report Generation:** Produces augmented summaries that merge **internal community sentiment** with **external narratives**.  

This paves the way for **holistic reporting** that is not only platform-specific but also globally aware.  


<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-34-57" src="https://github.com/user-attachments/assets/0c6141af-9dc9-462b-831e-477576652cb0" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-35-02" src="https://github.com/user-attachments/assets/886f5a78-3928-4489-8f7a-2ea49a060a66" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-35-14" src="https://github.com/user-attachments/assets/511748d6-bc2a-4742-bd13-934ceb03ff57" />


---

### 5. Neo4j Relationship Modeling  
I architected a **graph-based data model** using **Neo4j AuraDB Free Instance** to capture the intricate relationships between users, posts, comments, and topics.  

- **Graph Population:**  
  - **Clusters → Platforms (YouTube, Reddit)**  
  - **Clusters → Posts → Comments**  
  - **Users → Posts → Engagement Metrics**  
- **Advanced Queries Enabled:**  
  - Detecting **cross-platform influencers** who shape narratives across Reddit and YouTube simultaneously.  
  - Identifying **hidden community structures** and micro-communities.  
  - Mapping **thematic overlaps** where user-generated clusters converge on similar discussions.  
- **Scalability:** The graph model allows easy scaling into **temporal graphs**, enabling trend tracking over time.  

This design makes it possible to move beyond static analysis into **relationship-driven insights**.  

<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-41-03" src="https://github.com/user-attachments/assets/c3736173-6a6c-4e30-8e6e-2a2efd1ddcbf" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-41-12" src="https://github.com/user-attachments/assets/3cff1866-b19b-4ac1-8035-f82e0de1d71d" />

<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-41-43" src="https://github.com/user-attachments/assets/1e24cb70-4d2c-4b8a-8247-acb4c0870f1d" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-42-03" src="https://github.com/user-attachments/assets/76271283-d2a2-43c3-b8a0-2878bbe29a93" />
<img width="1920" height="1080" alt="Screenshot From 2025-09-11 02-42-24" src="https://github.com/user-attachments/assets/266639ab-041f-4a93-8191-1d1f728311b8" />





---

## API Endpoints

| Method | Endpoint                 | Description                                                  |
| :----- | :----------------------- | :----------------------------------------------------------- |
| `POST` | `/dashboard`             | Aggregated payload for dashboard visualizations.             |
| `POST` | `/platform-distribution` | Retrieves post distribution data.                            |
| `POST` | `/engagement-trends`     | Returns engagement metrics over time.                        |
| `POST` | `/top-posts`             | Fetches highest-performing posts by engagement.              |
| `POST` | `/influential-users`     | Identifies influential users.                                |
| `POST` | `/trending-keywords`     | Extracts trending keywords using AI.                         |
| `POST` | `/posts-histogram`       | Provides histogram of post frequency.                        |
| `POST` | `/trending-comments`     | Fetches high-engagement comments.                            |
| `POST` | `/sentiment-trends`      | Retrieves sentiment trends over time.                        |
| `POST` | `/cluster`               | Runs clustering algorithm.                                   |
| `POST` | `/cluster-chat`          | Conversational queries on clustered data.                    |
| `POST` | `/populate-database`     | Populates Neo4j with social media data.                      |
| `POST` | `/clear-database`        | Clears Neo4j database.                                       |
| `GET`  | `/cluster-info`          | Metadata about clusters in Neo4j.                            |
| `GET`  | `/topic-info`            | Fetches information on topics and prevalence.                |
| `POST` | `/neo4j-chat`            | Conversational queries against Neo4j.                        |

---

## Getting Started

Follow these steps to set up and run the application on your local machine.

### Prerequisites

-   **Node.js & npm** (or yarn) for the Next.js frontend.
-   **Python 3.8+ & pip** for the FastAPI backend.
-   **Neo4j Desktop** or a cloud-based Neo4j AuraDB instance.
-   A **Groq API key** for AI features.

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/buddhadevyash/insights-plus-simppl-task.git
    cd insights-plus-simppl-task
    ```

2. **Backend Setup**:
    - Navigate to the `backend` directory.
    - Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    - Create a `.env` file in the `backend/app` directory and add your credentials:
        ```env
        GROQ_API_KEY="your_groq_api_key"
        NEO4J_URI="your_neo4j_bolt_uri"
        NEO4J_USER="your_neo4j_username"
        NEO4J_PASSWORD="your_neo4j_password"
        ```

3. **Frontend Setup**:
    - Navigate to the `frontend` directory.
    - Install the required npm packages:
        ```bash
        npm install
        ```

---

## Usage

1. **Start the backend server**:
    - From the `backend` directory, run:
        ```bash
        uvicorn app.main:app --reload
        ```
    - The API will be available at `http://localhost:8000`.

2. **Start the frontend development server**:
    - From the `frontend` directory, run:
        ```bash
        npm run dev
        ```

3. **Access the application**:
    - Open your web browser and navigate to `http://localhost:3000`.

4. **Populate the Neo4j Database**:
    - Before you can see graph-based data, you must populate your Neo4j instance. Send a `POST` request to the `/populate-database` endpoint using a tool like `curl` or Postman:
        ```bash
        curl -X POST http://localhost:8000/populate-database
        ```
    - You can now explore the dashboard, interact with visualizations, and use the AI chat to analyze your social media data.


## Project Structure

1.  **Backend Structure**:
    ```
    backend/
    ├── app/
    │   ├── main.py              # FastAPI entry point
    │   ├── data/                # Data ingestion
    │   ├── embedded_json_data/  # Pre-computed embeddings
    │   │   ├── comments_reddit.json
    │   │   ├── comments_youtube.json
    │   │   ├── posts_reddit.json
    │   │   └── posts_youtube.json
    │   └── json_data/           # Raw sample JSON data
    ├── pyproject.toml
    └── requirements.txt
    ```

2.  **Frontend Structure**:
    ```
    frontend/
    ├── public/
    ├── src/
    │   ├── app/
    │   │   ├── api/news/route.js
    │   │   ├── chatbot/         # Chat UI and visualizations
    │   │   │   ├── chat-ui.jsx
    │   │   │   ├── page.jsx
    │   │   │   ├── ReportDisplay.jsx
    │   │   │   └── Visualization.jsx
    │   │   ├── components/dashboard/
    │   │   │   └── line-chart.tsx
    │   │   ├── visualize/page.jsx
    │   │   └── news/page.jsx
    │   ├── globals.css
    │   ├── layout.jsx
    │   └── page.jsx
    └── package.json
    ```

3.  **Root Directory**:
    ```
    insights-plus-simppl-task/
    └── README.md
    ```




