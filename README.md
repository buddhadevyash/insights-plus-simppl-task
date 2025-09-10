# Social Media Insights Dashboard & AI Analyst

This AI-driven web application provides a comprehensive solution for **analyzing and visualizing social media data** from YouTube and Reddit. It transforms raw data into actionable intelligence, empowering users to uncover insights, monitor engagement, and interact with data using natural language queries.  

The project leverages a modern and scalable architecture with:  
- **Next.js** for a responsive frontend.  
- **FastAPI** with Python for a high-performance backend.  
- **Neo4j** for graph-based data modeling.  
- **Groq API** for cutting-edge AI-powered analysis.  

---

# Technical System Overview

This project integrates **conversational AI, clustering, interactive dashboards, graph-based data modeling, and external data augmentation** to deliver advanced insights from YouTube and Reddit datasets. Below is a detailed breakdown of the components I designed.

---

### 1. Conversational AI–Driven Smart Reporting  
I designed a **conversational analytics engine** that transforms raw social media data into **inferential visualizations and reports** based on natural language queries.  

- **Embedding Generation:** Used `all-MiniLM-L6-v2` to convert unstructured text (posts, comments, discussions) into semantic embeddings.  
- **Similarity Search:** Implemented **cosine similarity** with **KNN (k=8)** retrieval to extract the most contextually relevant posts.  
- **Context Optimization:** Limited query expansion to the **≤8000 token cap** of Groq’s free tier by dynamically pruning low-confidence results.  
- **Reporting Framework:**  
  - **Summary Reports** → key engagement metrics, sentiment distributions, and posting frequencies.  
  - **Detailed Drilldowns** → fine-grained insights at the post/comment level.  
  - **Key Metrics Extraction** → automatically highlights top-performing entities.  
- **Inference Engine:** Integrated **OpenAI OSS 20B** on Groq, balancing performance with low-latency response times.  

This design ensures scalability with larger context windows when upgrading beyond free-tier Groq limits.  

![Smart Report](images/smart_report.png)

---

### 2. Cluster Querying & Tag-Based Exploration  
I developed a **semantic clustering pipeline** to automatically group related content into coherent themes, enabling both visualization and direct query access.  

- **Clustering:** Applied **KMeans** on embeddings to form **top-k thematic clusters**, capturing high-level discussion topics.  
- **Cluster Querying:** Implemented a `@ClusterName` tagging system, inspired by WhatsApp mentions, for easy referencing of specific clusters.  
- **Visualization:** Generated **cluster visualizations** to observe data point distributions, intra-cluster cohesion, and inter-cluster separation.  
- **Cluster Labeling:** Leveraged **LLaMA 8.1-Instant** to automatically assign descriptive, human-readable names to clusters (e.g., *AI Research Trends*, *Gaming Hardware*, *Community Feedback*).  

This enables me to move seamlessly between **macro-level trend discovery** and **micro-level cluster drilldowns**.  

![Cluster Querying](images/cluster_query.png)

---

### 3. Dynamic Real-Time Dashboard  
I implemented a **real-time interactive dashboard** to centralize insights and provide at-a-glance intelligence.  

- **Data Refreshing:** Designed for **dynamic updates** as new social data is ingested.  
- **Smart Filters:** Added filters for platform (YouTube/Reddit), sentiment range, engagement thresholds, and time periods.  
- **Leaderboards:** Automatically generates rankings of **top posts, authors, and clusters** to surface key contributors and discussions.  
- **Visualization Suite:** Includes engagement timelines, sentiment dynamics, keyword trends, and posting velocity histograms.  

This dashboard acts as a **command center**, making complex analysis accessible in a single interface.  

![Dashboard](images/dashboard.png)

---

### 4. News API Integration (Future Scope)  
As a future extension, I scoped out **News API integration** to contextualize platform-specific insights with **external events and real-world signals**.  

- **Embedding Augmentation:** External articles will be embedded alongside social data to enrich cluster context.  
- **Cross-Domain Correlation:** Enables tracing correlations between trending news and social platform engagement spikes.  
- **Optimal Report Generation:** Produces augmented summaries that merge **internal community sentiment** with **external narratives**.  

This paves the way for **holistic reporting** that is not only platform-specific but also globally aware.  

![News API](images/news_api.png)

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

![Neo4j Graph](images/neo4j_graph.png)

---

## Tech Stack

- **Frontend**: Next.js  
- **Backend**: FastAPI (Python)  
- **Database**: Neo4j  
- **AI/LLM**: Groq API  

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

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Backend Setup**:
    -   Navigate to the `backend` directory.
    -   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    -   Create a `.env` file in the `backend/app` directory and add your credentials:
        ```env
        GROQ_API_KEY="your_groq_api_key"
        NEO4J_URI="your_neo4j_bolt_uri"
        NEO4J_USER="your_neo4j_username"
        NEO4J_PASSWORD="your_neo4j_password"
        ```

3.  **Frontend Setup**:
    -   Navigate to the `frontend` directory.
    -   Install the required npm packages:
        ```bash
        npm install
        ```

---

## Usage

1.  **Start the backend server**:
    -   From the `backend` directory, run:
        ```bash
        uvicorn app.main:app --reload
        ```
    -   The API will be available at `http://localhost:8000`.

2.  **Start the frontend development server**:
    -   From the `frontend` directory, run:
        ```bash
        npm run dev
        ```

3.  **Access the application**:
    -   Open your web browser and navigate to `http://localhost:3000`.

4.  **Populate the Neo4j Database**:
    -   Before you can see graph-based data, you must populate your Neo4j instance. Send a `POST` request to the `/populate-database` endpoint using a tool like `curl` or Postman:
        ```bash
        curl -X POST http://localhost:8000/populate-database
        ```
    -   You can now explore the dashboard, interact with visualizations, and use the AI chat to analyze your social media data.

---

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


