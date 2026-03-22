# 📊 Hybrid RAG + Data Agent System

A Streamlit-based question-answering application that combines **Retrieval-Augmented Generation (RAG)** for semantic search with a **Pandas DataFrame Agent** for structured data analysis. Upload PDFs, text files, and CSVs — then ask questions in plain English.

---

## ✨ Features

- 🤖 **Intelligent Query Routing** — automatically classifies questions as aggregation (statistical) or semantic search
- 🔍 **Semantic Search (RAG)** — uses FAISS vector store + OpenAI embeddings to retrieve relevant document passages
- 📊 **Pandas Agent** — runs live Python/Pandas code to answer numerical and aggregation queries on CSV data
- 📄 **Multi-format Support** — PDF, TXT, and CSV files
- 🏷️ **Auto Document Classification** — PDFs are automatically tagged (Invoice, Contract, Bank Statement, etc.)
- 📚 **Source Attribution** — every answer cites which document it came from
- ⚡ **Session Management** — process new file sets without restarting the app

---

## 🗂️ Project Structure

```
├── app.py               # Streamlit frontend
├── main_dynamic.py      # Core backend: file processing, RAG pipeline, Pandas agent
├── .env                 # Local environment variables (not committed)
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `langchain` | LLM orchestration |
| `langchain-openai` | OpenAI LLM + embeddings |
| `langchain-community` | FAISS, document loaders |
| `langchain-experimental` | Pandas DataFrame Agent |
| `faiss-cpu` | Vector similarity search |
| `pypdf` | PDF parsing |
| `pandas` | CSV / tabular data |
| `python-dotenv` | Local env variable loading |

### 3. Set your OpenAI API key

**For local development**, create a `.env` file in the project root:

```
OPENAI_KEY=sk-...your-key-here...
```

**For Streamlit Cloud deployment**, add the key in the app's Secrets settings:

```toml
# .streamlit/secrets.toml
OPENAI_KEY = "sk-...your-key-here..."
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. **Upload Files** — use the sidebar to upload one or more PDF, TXT, or CSV files
2. **Process** — click **Process Files** to build the knowledge base and agents
3. **Ask Questions** — type any question in the main input box
4. **View Results** — the answer is displayed with source attribution; optionally toggle retrieved context

---

## 🧠 How It Works

```
User Question
      │
      ▼
 Query Classifier (LLM)
      │
   ┌──┴──┐
   │     │
Aggregation  Semantic Search
   │         │
Pandas     FAISS
Agent      Vector DB
   │         │
   └──┬──────┘
      ▼
   Answer + Sources
```

### Aggregation queries
Questions involving `sum`, `average`, `count`, `max`, `min`, etc. are routed to a **LangChain Pandas DataFrame Agent**, which generates and executes Python code against your CSV data.

### Semantic search queries
All other questions are handled by the **RAG pipeline**: the query is embedded, the top-3 most similar document chunks are retrieved from FAISS, and an LLM generates an answer grounded in that context.

---

## 💡 Example Queries

**Aggregation (CSV data)**
- *What is the average spending score?*
- *How many records are in the dataset?*
- *What is the total revenue?*

**Semantic search (PDF/TXT)**
- *What is the refund policy?*
- *Summarize the contract terms.*
- *What are the key points of this document?*

---

## 🌐 Deploying to Streamlit Cloud

1. Push your code to a public GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. Add `OPENAI_KEY` under **App Settings → Secrets**
4. Deploy — no other configuration required

---

## 🔒 Security Notes

- Never commit your `.env` file or API keys to version control
- Add `.env` to your `.gitignore`
- The Pandas agent uses `allow_dangerous_code=True` — only deploy in trusted environments

---

## 📄 License

MIT
