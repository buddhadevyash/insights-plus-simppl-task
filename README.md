# Social Media Insights Dashboard & AI Analyst

This powerful, AI-driven web application offers a comprehensive solution for analyzing and visualizing social media data from YouTube and Reddit. It leverages a sophisticated tech stack, including **Next.js** for a responsive frontend, **Python** with **FastAPI** for a high-performance backend, **Neo4j** for insightful graph-based data modeling, and the **Groq API** for cutting-edge AI-powered analysis.

The platform transforms raw social media data into actionable intelligence, allowing users to uncover deep insights, track community engagement, and have a natural language conversation with their data.

---

## Features in Detail

This application is packed with features designed to provide a 360-degree view of your social media landscape.

### 1. Cross-Platform Analysis
Go beyond siloed data views. The dashboard allows you to seamlessly filter and analyze data from YouTube, Reddit, or both platforms combined. This enables powerful comparative analysis, helping you answer critical questions like:
- Does our content perform better on YouTube or Reddit?
- Where are the most engaged conversations happening?
- How do topics and sentiment differ between the two platforms?
This feature provides the context needed to tailor your content strategy for each specific audience.

### 2. Rich & Interactive Data Visualizations
A suite of dynamic charts and graphs brings your data to life, providing at-a-glance insights into performance and trends. Each visualization is designed to answer a specific business question:

-   **Platform Distribution**: A clear pie chart that shows the volume share of posts from each platform. Instantly understand which platform is more active and where the bulk of your data originates.
-   **Engagement Trends**: A time-series line chart that tracks key engagement metrics (likes, comments, upvotes) over your selected date range. This helps you identify peak engagement periods, spot seasonal trends, and measure the impact of specific campaigns or events.
-   **Top Performing Posts**: A ranked list of your most successful content. This component doesn't just show a title; it provides the post content, author, platform, direct URL, and a calculated engagement score, allowing you to quickly analyze what makes top content resonate with your audience.
-   **Most Influential Users**: Discover the key players in your community. This feature uses a weighted algorithm that considers total engagement, post count, and average engagement per post to identify true influencers, not just frequent posters.
-   **AI-Powered Trending Keywords**: Using the Groq API, this feature moves beyond simple word clouds. It analyzes the text from all posts and comments to extract the most relevant and statistically significant topics and keywords, helping you stay ahead of conversations and understand what your audience truly cares about.
-   **Posts Histogram**: A bar chart that visualizes content velocity. It shows the number of posts made per time period (e.g., per week), allowing you to identify content scheduling patterns and periods of high or low activity.
-   **Trending Comments**: Dive deep into audience conversations. This feature identifies and ranks the comments with the highest engagement, offering a direct window into popular opinions, key questions, and the most impactful interactions within your community.
-   **Sentiment Analysis Trends**: A multi-line chart that acts as a barometer for community health. It tracks the average positive, negative, and neutral sentiment scores over time for each platform, allowing you to monitor audience perception and the emotional response to your content.

### 3. AI-Powered Thematic Clustering
This advanced feature uses machine learning to automatically group semantically similar posts and comments into distinct clusters. The process involves:
1.  Converting text content into numerical vector embeddings.
2.  Using a KMeans clustering algorithm to group related content.
3.  Sending a sample of content from each cluster to the Groq LLM to generate a short, descriptive, and human-readable name (e.g., "AI Technology & ChatGPT," "Community Feedback & Issues").
This turns thousands of unstructured data points into organized, understandable themes, revealing the core topics of discussion without manual effort.

### 4. Conversational AI Chat
Engage in a natural language conversation with your data. This feature acts as your personal AI data analyst. You can ask specific questions about the clustered data, such as:
-   "What is the overall sentiment in the 'Community Feedback & Issues' cluster?"
-   "Which users are most active in discussions about AI?"
-   "Summarize the key points from the 'Gaming Hardware' cluster."
The AI provides detailed, context-aware answers based on the underlying data, making deep analysis accessible to everyone, regardless of their technical skills.

### 5. Graph-Based Data Modeling with Neo4j
The application's foundation is a powerful Neo4j graph database. Instead of storing data in rigid tables, it maps out the complex relationships between entities: `Users`, `Posts`, `Comments`, and `Topics`. This connected structure enables sophisticated queries that are nearly impossible with traditional databases, such as:
-   Identifying influential users who frequently post about a specific topic.
-   Finding posts that generated comments from users across multiple platforms.
-   Discovering hidden community structures and interaction patterns.

### 6. Scalable & Efficient Backend
Built with FastAPI and its asynchronous capabilities, the backend is designed for high performance and scalability. It can handle multiple concurrent requests for data and visualizations without blocking, ensuring a fast, fluid, and responsive user experience even when dealing with large datasets and complex AI-driven analyses.

---

## Tech Stack

-   **Frontend**: **Next.js** - A modern React framework for building fast, server-rendered, and user-friendly web applications.
-   **Backend**: **Python** with **FastAPI** - A high-performance web framework for building robust and scalable APIs with Python.
-   **Database**: **Neo4j** - A native graph database ideal for modeling and querying highly connected social media data to uncover deep relationships.
-   **AI/LLM**: **Groq API** - Provides blazing-fast access to large language models for advanced capabilities like conversational chat, thematic clustering, and keyword extraction.

---

## API Endpoints

The backend exposes a comprehensive RESTful API. Here are the key endpoints:

| Method | Endpoint                 | Description                                                  |
| :----- | :----------------------- | :----------------------------------------------------------- |
| `POST` | `/dashboard`             | Fetches an aggregated payload for all main dashboard visualizations. |
| `POST` | `/platform-distribution` | Retrieves post distribution data between YouTube and Reddit. |
| `POST` | `/engagement-trends`     | Gets engagement metrics over a specified time period.        |
| `POST` | `/top-posts`             | Fetches the highest-performing posts based on engagement scores. |
| `POST` | `/influential-users`     | Identifies key users based on their engagement and activity. |
| `POST` | `/trending-keywords`     | Extracts and returns trending keywords using AI.             |
| `POST` | `/posts-histogram`       | Provides a histogram of post frequency over time.            |
| `POST` | `/trending-comments`     | Fetches the top-performing comments based on engagement.     |
| `POST` | `/sentiment-trends`      | Retrieves sentiment scores (positive, negative, neutral) over time. |
| `POST` | `/cluster`               | Runs the AI-powered clustering algorithm on the dataset.     |
| `POST` | `/cluster-chat`          | Enables conversational queries about the clustered data.     |
| `POST` | `/populate-database`     | Populates the Neo4j database with social media data.         |
| `POST` | `/clear-database`        | Clears all nodes and relationships from the Neo4j database.  |
| `GET`  | `/cluster-info`          | Retrieves metadata about the clusters stored in Neo4j.       |
| `GET`  | `/topic-info`            | Fetches information about topics and their prevalence.       |
| `POST` | `/neo4j-chat`            | A conversational endpoint to interact with Neo4j using natural language. |

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
