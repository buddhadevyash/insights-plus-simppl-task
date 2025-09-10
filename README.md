# Social Media Insights Dashboard & AI Analyst

This AI-driven web application provides a comprehensive solution for **analyzing and visualizing social media data** from YouTube and Reddit. It transforms raw data into actionable intelligence, empowering users to uncover insights, monitor engagement, and interact with data using natural language queries.  

The project leverages a modern and scalable architecture with:  
- **Next.js** for a responsive frontend.  
- **FastAPI** with Python for a high-performance backend.  
- **Neo4j** for graph-based data modeling.  
- **Groq API** for cutting-edge AI-powered analysis.  

---

## Key Features

### 1. Cross-Platform Analysis  
Unify insights across YouTube and Reddit. The dashboard enables:  
- Comparative content performance across platforms.  
- Identification of where engagement is strongest.  
- Differences in sentiment and trending topics.  

This enables smarter strategy design tailored for each audience.  

### 2. Rich & Interactive Visualizations  
Dynamic and interactive charts provide at-a-glance insights:  
- **Platform Distribution** – Understand post share by platform.  
- **Engagement Trends** – Track likes, comments, and upvotes over time.  
- **Top Performing Posts** – Analyze top posts with engagement scores, authors, and links.  
- **Most Influential Users** – Identify users with genuine influence, not just high activity.  
- **AI-Powered Trending Keywords** – Extract significant topics using Groq API.  
- **Posts Histogram** – Reveal posting velocity and scheduling patterns.  
- **Trending Comments** – Surface high-engagement audience opinions.  
- **Sentiment Analysis Trends** – Monitor sentiment dynamics over time.  

### 3. AI-Powered Thematic Clustering  
Automatically group thousands of posts and comments into coherent themes:  
1. Convert content into embeddings.  
2. Apply KMeans clustering.  
3. Use Groq LLM to label clusters with descriptive names (e.g., *“AI Technology & ChatGPT”*).  

This converts unstructured data into digestible insights.  

### 4. Conversational AI Chat  
Ask natural language questions such as:  
- “What is the sentiment in the *Community Feedback & Issues* cluster?”  
- “Which users are most active in AI-related discussions?”  
- “Summarize the *Gaming Hardware* cluster.”  

The AI returns context-rich answers backed by your data.  

### 5. Graph-Based Data Modeling with Neo4j  
Neo4j maps complex relationships across `Users`, `Posts`, `Comments`, and `Topics`, enabling advanced queries such as:  
- Identifying influencers on niche topics.  
- Detecting cross-platform conversations.  
- Discovering hidden community structures.  

### 6. Scalable & Efficient Backend  
The backend, built with **FastAPI**, delivers asynchronous, high-performance APIs capable of handling large datasets and concurrent requests.  

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

A high-level overview of the project's architecture:
insights-plus-simppl-task/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── data/                # Data ingestion
│   │   ├── embedded_json_data/  # Pre-computed embeddings
│   │   │   ├── comments_reddit.json
│   │   │   ├── comments_youtube.json
│   │   │   ├── posts_reddit.json
│   │   │   └── posts_youtube.json
│   │   └── json_data/           # Raw sample JSON data
│   ├── pyproject.toml
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/news/route.js
│   │   │   ├── chatbot/         # Chat UI and visualizations
│   │   │   │   ├── chat-ui.jsx
│   │   │   │   ├── page.jsx
│   │   │   │   ├── ReportDisplay.jsx
│   │   │   │   └── Visualization.jsx
│   │   │   ├── components/dashboard/
│   │   │   │   └── line-chart.tsx
│   │   │   ├── visualize/page.jsx
│   │   │   └── news/page.jsx
│   │   ├── globals.css
│   │   ├── layout.jsx
│   │   └── page.jsx
│   └── package.json
│
└── README.md

