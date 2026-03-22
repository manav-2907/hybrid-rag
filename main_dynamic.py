import pandas as pd
import logging
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
# Load API key: try .env first (local dev), then Streamlit secrets (cloud deployment)
load_dotenv()
api_key = os.getenv("OPENAI_KEY")

if api_key:
    logger.info("Using API key from .env file")
else:
    # Fallback to Streamlit secrets for cloud deployment
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_KEY")
        logger.info("Using API key from Streamlit secrets")
    except:
        pass

if not api_key:
    raise EnvironmentError("OPENAI_KEY not found. Add it to .env file or Streamlit secrets.")

# ─── Global state (will be populated by process_uploaded_files) ──────────────
vector_db = None
pandas_agents = {}
dfs = {}
llm = None

# ─── Initialize LLM ───────────────────────────────────────────────────────────
def initialize_llm():
    """Initialize the LLM (called once)"""
    global llm
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=api_key,
        )
        logger.info("LLM initialized")
    return llm

# ─── Process uploaded files ───────────────────────────────────────────────────
def process_uploaded_files(file_paths: List[str]) -> Dict:
    """
    Process uploaded files and build vector DB + pandas agents
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        Dictionary with processing statistics
    """
    global vector_db, pandas_agents, dfs, llm
    
    # Initialize LLM
    llm = initialize_llm()
    
    pdf_docs = []
    text_docs = []
    csv_docs = []
    dfs = {}
    pandas_agents = {}
    
    pdf_count = 0
    text_count = 0
    csv_count = 0
    
    logger.info(f"Processing {len(file_paths)} files...")
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        try:
            # ─── Process PDF ──────────────────────────────────────────────────
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text = " ".join([p.page_content for p in pages])
                text_lower = text.lower()
                
                # Auto-classify document type
                if "court" in text_lower and "challan" in text_lower:
                    doc_type = "Challan_Receipt"
                elif "credit" in text_lower and "statement" in text_lower:
                    doc_type = "Bank_Statement"
                elif "ticket" in text_lower:
                    doc_type = "Ticket"
                elif "invoice" in text_lower:
                    doc_type = "Invoice"
                elif "contract" in text_lower or "agreement" in text_lower:
                    doc_type = "Contract"
                else:
                    doc_type = "General_Document"
                
                pdf_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_name, "type": doc_type},
                    )
                )
                pdf_count += 1
                logger.info(f"✅ Loaded PDF: {file_name} → {doc_type}")
            
            # ─── Process CSV ──────────────────────────────────────────────────
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                
                # Store dataframe for pandas agent
                dfs[file_name] = df
                
                # Also add to documents for RAG
                for _, row in df.iterrows():
                    content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                    csv_docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": file_name, "type": "csv_row"},
                        )
                    )
                
                csv_count += 1
                logger.info(f"✅ Loaded CSV: {file_name} ({len(df)} rows)")
            
            # ─── Process TXT ──────────────────────────────────────────────────
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                text_docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_name, "type": "text_file"},
                    )
                )
                text_count += 1
                logger.info(f"✅ Loaded TXT: {file_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {file_name}: {e}")
    
    # ─── Build vector database ────────────────────────────────────────────────
    documents = pdf_docs + csv_docs + text_docs
    total_docs = len(documents)
    
    if total_docs == 0:
        logger.warning("No documents were processed")
        return {
            "total_docs": 0,
            "pdf_count": 0,
            "text_count": 0,
            "csv_count": 0
        }
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {total_docs} documents")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    logger.info("✅ FAISS vector database created")
    
    # ─── Create pandas agents ─────────────────────────────────────────────────
    for name, df in dfs.items():
        pandas_agents[name] = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            prefix=f"""
You are working with a pandas dataframe named 'df' from the file '{name}'.

Available columns (use EXACTLY as shown):
{list(df.columns)}

CRITICAL: You have ONLY ONE TOOL: python_repl_ast

Always use this format:
Action: python_repl_ast
Action Input: <your pandas code>

Examples:
- df['column_name'].mean()
- df['column_name'].sum()
- len(df)
- df['column_name'].value_counts()

Rules:
- Use EXACT column names from the list above
- If a column is not in the list, say "Column not found"
- Always return numerical results when doing calculations
""",
        )
    
    logger.info(f"✅ Created {len(pandas_agents)} pandas agents")
    
    return {
        "total_docs": total_docs,
        "pdf_count": pdf_count,
        "text_count": text_count,
        "csv_count": csv_count,
        "chunks": len(chunks)
    }


# ─── Query classifier ─────────────────────────────────────────────────────────
def classify_query(query: str) -> str:
    """
    Classify query as 'aggregation' or 'semantic_search'
    """
    if llm is None:
        initialize_llm()
    
    prompt = f"""Classify this question as either 'aggregation' or 'semantic_search'.

Aggregation = requires computing sum, count, average, max, min, or any statistical operation
Semantic search = requires retrieving specific facts, definitions, or passages from documents

Question: {query}

Answer with ONLY one word: aggregation or semantic_search"""

    try:
        result = llm.invoke(prompt).content.strip().lower()
        if result not in ("aggregation", "semantic_search"):
            logger.warning(f"Unexpected classifier output '{result}', defaulting to semantic_search")
            return "semantic_search"
        logger.info(f"Query classified as: {result}")
        return result
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return "semantic_search"


# ─── CSV selector ─────────────────────────────────────────────────────────────
def select_csv(query: str) -> str:
    """
    Select the best CSV file for the query
    """
    if not dfs:
        return None
    
    if len(dfs) == 1:
        return list(dfs.keys())[0]
    
    if llm is None:
        initialize_llm()
    
    csv_info = {name: list(df.columns) for name, df in dfs.items()}
    prompt = f"""Given these CSV files and their columns:
{csv_info}

Question: {query}

Which CSV file should be used? Respond with ONLY the filename."""

    try:
        selected = llm.invoke(prompt).content.strip()
        if selected in pandas_agents:
            logger.info(f"Selected CSV: {selected}")
            return selected
        else:
            logger.warning(f"LLM selected unknown CSV '{selected}', using first available")
            return list(pandas_agents.keys())[0]
    except Exception as e:
        logger.error(f"CSV selection failed: {e}")
        return list(pandas_agents.keys())[0]


# ─── Main Q&A function ────────────────────────────────────────────────────────
def ask_question(query: str, doc_type_filter: str = None) -> dict:
    """
    Answer questions using the appropriate pipeline
    """
    if llm is None:
        initialize_llm()
    
    # Classify the query
    intent = classify_query(query)
    
    # ── Aggregation branch ────────────────────────────────────────────────────
    if intent == "aggregation":
        if not pandas_agents:
            return {
                "answer": "No CSV files were uploaded. I can only answer questions about text documents.",
                "sources": [],
                "context": "",
            }
        
        selected_csv = select_csv(query)
        agent = pandas_agents[selected_csv]
        
        try:
            response = agent.invoke(query)
            return {
                "answer": response["output"],
                "sources": [f"CSV: {selected_csv} (Pandas Agent)"],
                "context": "Computed from structured data",
            }
        except Exception as e:
            logger.error(f"Pandas agent failed: {e}")
            return {
                "answer": f"I encountered an error processing your aggregation query: {str(e)}",
                "sources": [f"CSV: {selected_csv}"],
                "context": "",
            }
    
    # ── Semantic search branch ────────────────────────────────────────────────
    if vector_db is None:
        return {
            "answer": "No documents have been processed yet. Please upload some files first.",
            "sources": [],
            "context": "",
        }
    
    try:
        search_kwargs = {"k": 3}
        
        # Apply metadata filter if specified
        if doc_type_filter:
            logger.info(f"Applying filter: type='{doc_type_filter}'")
            retriever = vector_db.as_retriever(
                search_kwargs={**search_kwargs, "filter": {"type": doc_type_filter}}
            )
            results = retriever.invoke(query)
        else:
            results = vector_db.similarity_search(query, **search_kwargs)
        
        if not results:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                "sources": [],
                "context": "",
            }
        
        context = "\n\n".join([doc.page_content for doc in results])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer (be concise and based only on the context):"""

        response = llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": [doc.metadata for doc in results],
            "context": context,
        }
        
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        return {
            "answer": f"I encountered an error: {str(e)}",
            "sources": [],
            "context": "",
        }


# ─── Clear session ────────────────────────────────────────────────────────────
def clear_session():
    """Clear all global state"""
    global vector_db, pandas_agents, dfs
    vector_db = None
    pandas_agents = {}
    dfs = {}
    logger.info("Session cleared")
