
# streamlit run app.py
"""
Nexora College Help Desk
------------------------
A Retrieval-Augmented Generation (RAG) chatbot that answers student
questions about admissions, timetables, fees, and rules — grounded
strictly in Nexora College's official handbook and circulars.
 
Modules:
    1. Handbook & Circular Ingestion (PDF + TXT)
    2. Chunking & Embedding
    3. Vector Store (FAISS, with optional save/load to disk)
    4. Retriever
    5. Student Chat Interface
"""
 
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
 
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
 
COLLEGE_NAME = "Nexora Institute of Technology"
KB_STORE_DIR = "nexora_kb_store"  # local folder where the FAISS index is persisted
SAMPLE_DOCS_DIR = "sample_docs"  # bundled demo handbook/circulars, shipped next to app.py
 
 
# ============================================================
# 1. Page config + configuration loading
# ============================================================
st.set_page_config(page_title=f"{COLLEGE_NAME} Help Desk",
                   page_icon="🎓",
                   layout="wide")
 
load_dotenv()
 
 
def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --nexora-primary: #1E3A8A;
            --nexora-accent: #D4A017;
        }
 
        /* Tighten the default top padding */
        .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }
 
        /* Header banner */
        .nexora-header {
            background: linear-gradient(90deg, var(--nexora-primary), #2E4FA8);
            padding: 1.1rem 1.5rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.2rem;
        }
        .nexora-header h1 {
            margin: 0;
            font-size: 1.5rem;
            color: white;
        }
        .nexora-header p {
            margin: 0.25rem 0 0 0;
            font-size: 0.92rem;
            color: #E7ECFB;
        }
 
        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.4rem 0.2rem;
        }
 
        /* Sidebar polish */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(0,0,0,0.06);
        }
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: var(--nexora-primary);
        }
 
        /* Keep chat input clean and pinned */
        [data-testid="stChatInput"] textarea {
            border-radius: 10px !important;
        }
 
        .nexora-badge {
            display: inline-block;
            background: #EEF2FF;
            color: var(--nexora-primary);
            border-radius: 999px;
            padding: 0.1rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
def get_config_value(key, default=None):
    """Read from environment (.env, local) or st.secrets (Streamlit Cloud)."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default
 
 
HF_TOKEN = get_config_value("HF_TOKEN")
MODEL_NAME = get_config_value("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
ADMIN_PASSCODE = get_config_value("ADMIN_PASSCODE", "")  # optional; blank = no lock
 
if not HF_TOKEN:
    st.error(
        "HF_TOKEN is missing. Add it to a local `.env` file, or on Streamlit "
        "Community Cloud go to **Settings -> Secrets** and add:\n\n"
        "```\nHF_TOKEN = \"hf_your_token\"\n```"
    )
    st.stop()
 
 
# ============================================================
# 2. Create clients/models once when the app starts
# ============================================================
@st.cache_resource(show_spinner=False)
def get_llm_client():
    return InferenceClient(provider="auto", token=HF_TOKEN)
 
 
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
 
 
client = get_llm_client()
embedding_model = get_embedding_model()
 
 
# ============================================================
# 3. Session state
# ============================================================
DEFAULTS = {
    "retriever": None,
    "chat_history": [],
    "status_message": "No knowledge base loaded yet. Ask an admin to upload the handbook and circulars.",
    "sources_text": "",
    "doc_registry": [],       # list of {"name":..., "category":..., "pages":...}
    "kb_built_at": None,
    "admin_unlocked": ADMIN_PASSCODE == "",  # if no passcode set, admin tab is open
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
 
 
# ============================================================
# 4. Build the vector database from uploaded documents
# ============================================================
def _load_single_file(file_path: Path, category: str, original_name: str):
    """Load one PDF or TXT file into LangChain Document objects, tagged with metadata."""
    suffix = file_path.suffix.lower()
 
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        pages = loader.load()
    else:
        return []
 
    for page in pages:
        page.metadata["source_name"] = original_name
        page.metadata["category"] = category
 
    return pages
 
 
def _finalize_knowledge_base(documents, new_registry_entries, append, progress_bar, source_label="uploaded files"):
    """
    Shared tail-end pipeline: split into chunks -> embed -> build/merge FAISS
    -> create retriever -> update session state and status message.
    Used by both the upload-based builder and the bundled sample-doc loader.
    """
    if not documents:
        st.session_state.status_message = f"No readable content was found in the {source_label}."
        return
 
    progress_bar.progress(0.45, text="Splitting documents into chunks")
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
 
    if not chunks:
        st.session_state.status_message = f"The {source_label} were loaded, but no text chunks were created."
        return
 
    progress_bar.progress(0.60, text="Creating embeddings")
 
    new_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
 
    if append and st.session_state.get("_vector_store") is not None:
        st.session_state["_vector_store"].merge_from(new_store)
        vector_store = st.session_state["_vector_store"]
        st.session_state.doc_registry.extend(new_registry_entries)
    else:
        vector_store = new_store
        st.session_state.doc_registry = new_registry_entries
 
    st.session_state["_vector_store"] = vector_store
 
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
 
    st.session_state.retriever = retriever
    if not append:
        st.session_state.chat_history = []
    st.session_state.sources_text = ""
    st.session_state.kb_built_at = datetime.now().strftime("%Y-%m-%d %H:%M")
 
    progress_bar.progress(1.0, text="Knowledge base ready")
 
    total_docs = len(st.session_state.doc_registry)
    total_pages = sum(entry["pages"] for entry in st.session_state.doc_registry)
    st.session_state.status_message = (
        f"Knowledge base ready for {COLLEGE_NAME}.\n\n"
        f"Documents indexed: {total_docs}\n"
        f"Pages/sections loaded: {total_pages}\n"
        f"Chunks in this build: {len(chunks)}\n"
        f"Last updated: {st.session_state.kb_built_at}"
    )
 
 
def build_knowledge_base(handbook_files, circular_files, other_files, append=False):
    """
    Load handbook, circular, and other supporting documents (as uploaded by
    the admin) -> split into chunks -> create embeddings -> build (or
    extend) a FAISS vector store -> create a retriever.
    """
    grouped = [
        ("Handbook", handbook_files or []),
        ("Circular", circular_files or []),
        ("Other", other_files or []),
    ]
 
    total_files = sum(len(files) for _, files in grouped)
    if total_files == 0:
        st.session_state.status_message = "Please upload at least one file (handbook, circular, or other)."
        return
 
    progress_bar = st.progress(0, text="Loading documents")
 
    try:
        documents = []
        new_registry_entries = []
        processed = 0
 
        with tempfile.TemporaryDirectory() as tmp_dir:
            for category, files in grouped:
                for uploaded_file in files:
                    tmp_path = Path(tmp_dir) / uploaded_file.name
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
 
                    loaded_pages = _load_single_file(tmp_path, category, uploaded_file.name)
                    documents.extend(loaded_pages)
 
                    new_registry_entries.append({
                        "name": uploaded_file.name,
                        "category": category,
                        "pages": len(loaded_pages),
                    })
 
                    processed += 1
                    progress_bar.progress(
                        min(0.35, 0.05 + (processed / max(total_files, 1)) * 0.30),
                        text=f"Loaded {uploaded_file.name} ({category})",
                    )
 
            _finalize_knowledge_base(documents, new_registry_entries, append, progress_bar,
                                      source_label="uploaded files")
 
    except Exception as error:
        st.session_state.status_message = f"Could not build the knowledge base.\n\nError: {error}"
 
    finally:
        progress_bar.empty()
 
 
def load_sample_knowledge_base(append=False):
    """
    Build the knowledge base from the bundled sample Nexora Institute of
    Technology handbook, circulars, and reference documents shipped in the
    sample_docs/ folder next to this app.
    Lets the app work end-to-end for a demo without needing any uploads.
    """
    sample_root = Path(SAMPLE_DOCS_DIR)
    category_folders = [
        ("Handbook", sample_root / "handbook"),
        ("Circular", sample_root / "circulars"),
        ("Other", sample_root / "other"),
    ]
 
    file_specs = []
    for category, folder in category_folders:
        if not folder.exists():
            continue
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() in (".pdf", ".txt", ".md"):
                file_specs.append((file_path, category))
 
    if not file_specs:
        st.session_state.status_message = (
            f"No bundled sample documents were found under '{SAMPLE_DOCS_DIR}/'. "
            "Make sure the sample_docs folder is deployed alongside app.py."
        )
        return
 
    progress_bar = st.progress(0, text="Loading bundled Nexora sample documents")
 
    try:
        documents = []
        new_registry_entries = []
 
        for index, (file_path, category) in enumerate(file_specs, start=1):
            loaded_pages = _load_single_file(file_path, category, file_path.name)
            documents.extend(loaded_pages)
 
            new_registry_entries.append({
                "name": file_path.name,
                "category": category,
                "pages": len(loaded_pages),
            })
 
            progress_bar.progress(
                min(0.35, 0.05 + (index / max(len(file_specs), 1)) * 0.30),
                text=f"Loaded {file_path.name} ({category})",
            )
 
        _finalize_knowledge_base(documents, new_registry_entries, append, progress_bar,
                                  source_label="bundled sample documents")
 
    except Exception as error:
        st.session_state.status_message = f"Could not load the sample knowledge base.\n\nError: {error}"
 
    finally:
        progress_bar.empty()
 
 
def save_knowledge_base():
    vector_store = st.session_state.get("_vector_store")
    if vector_store is None:
        st.session_state.status_message = "There is no knowledge base in memory to save."
        return
    try:
        vector_store.save_local(KB_STORE_DIR)
        st.session_state.status_message = (
            f"Knowledge base saved to disk at '{KB_STORE_DIR}'. "
            "It will auto-load next time the app starts."
        )
    except Exception as error:
        st.session_state.status_message = f"Could not save the knowledge base.\n\nError: {error}"
 
 
def load_knowledge_base_from_disk():
    if not Path(KB_STORE_DIR).exists():
        st.session_state.status_message = "No saved knowledge base was found on disk yet."
        return
    try:
        vector_store = FAISS.load_local(
            KB_STORE_DIR, embedding_model, allow_dangerous_deserialization=True
        )
        st.session_state["_vector_store"] = vector_store
        st.session_state.retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        st.session_state.status_message = f"Loaded saved knowledge base for {COLLEGE_NAME} from disk."
    except Exception as error:
        st.session_state.status_message = f"Could not load the saved knowledge base.\n\nError: {error}"
 
 
def clear_session():
    """Remove the vector database and clear the interface."""
    st.session_state.retriever = None
    st.session_state["_vector_store"] = None
    st.session_state.chat_history = []
    st.session_state.status_message = "No knowledge base loaded yet. Ask an admin to upload the handbook and circulars."
    st.session_state.sources_text = ""
    st.session_state.doc_registry = []
    st.session_state.kb_built_at = None
 
 
# ============================================================
# 5. Generate an answer using the retrieved document context
# ============================================================
def generate_answer(question, context, attempts=3):
    """
    Send the question and retrieved context to the hosted LLM.
    Retry temporary API failures before returning an error message.
    """
    system_prompt = (
        f"You are the official AI Help Desk assistant for {COLLEGE_NAME}. "
        "You answer student questions about admissions, timetables, fees, and rules "
        "using ONLY the supplied document context, which comes from the college's "
        "official handbook and circulars. "
        "Do not use outside knowledge, and do not guess or make up policies, dates, or amounts. "
        "If the answer is not clearly present in the context, respond exactly with: "
        f"\"I don't know based on the {COLLEGE_NAME} handbook and circulars currently available. "
        "Please check with the administration office.\" "
        "When the context includes a date, deadline, or fee amount, quote it exactly as written. "
        "If circulars and the handbook appear to conflict, mention the conflict and note that the "
        "more recent circular normally takes precedence, and advise the student to confirm with the office. "
        "Keep answers clear, concise, and student-friendly. "
        "After the answer, do not repeat the raw source tags — the app will display sources separately."
    )
 
    user_prompt = f"""
DOCUMENT CONTEXT:
{context}
 
STUDENT QUESTION:
{question}
 
ANSWER:
""".strip()
 
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
 
            answer = response.choices[0].message.content
 
            if not answer:
                raise ValueError("The model returned an empty response.")
 
            return answer.strip()
 
        except Exception as error:
            if attempt < attempts:
                time.sleep(2)
            else:
                return (
                    "The AI service is temporarily unavailable. "
                    f"Please try again.\n\nTechnical details: {error}"
                )
 
 
# ============================================================
# 6. Answer a single student question (retrieval + generation only —
#    no chat_history side effects, so the UI can control the sequencing
#    of "show question -> show spinner -> show answer")
# ============================================================
def get_answer_and_sources(question):
    """
    Retrieve relevant chunks and generate a grounded answer.
    Returns (answer_text, list_of_source_lines).
    """
    question = (question or "").strip()
 
    if not question:
        return "", []
 
    if st.session_state.retriever is None:
        return (
            f"The {COLLEGE_NAME} knowledge base hasn't been set up yet. "
            "Please ask an administrator to upload the handbook and circulars first.",
            [],
        )
 
    try:
        relevant_docs = st.session_state.retriever.invoke(question)
 
        if not relevant_docs:
            answer = (
                f"I don't know based on the {COLLEGE_NAME} handbook and circulars currently "
                "available. Please check with the administration office."
            )
            return answer, []
 
        context_parts = []
        source_lines = []
 
        for number, document in enumerate(relevant_docs, start=1):
            source_name = document.metadata.get(
                "source_name",
                Path(document.metadata.get("source", "Unknown document")).name,
            )
            category = document.metadata.get("category", "Document")
 
            # PyPDFLoader stores zero-based page numbers; TextLoader has none.
            page_number = document.metadata.get("page")
            readable_page = page_number + 1 if isinstance(page_number, int) else "N/A"
 
            context_parts.append(
                f"[Source {number}: {category} — {source_name}, page {readable_page}]\n"
                f"{document.page_content}"
            )
            source_lines.append(f"[{category}] {source_name} — page {readable_page}")
 
        context = "\n\n".join(context_parts)
        answer = generate_answer(question, context)
        return answer, source_lines
 
    except Exception as error:
        return f"I could not process that question. Please try again.\n\nError: {error}", []
 
 
# ============================================================
# 7. Build the Streamlit interface
# ============================================================
inject_custom_css()
 
kb_ready = st.session_state.retriever is not None
 
# ---------------- Sidebar: status + admin (everything non-chat lives here) ----------------
with st.sidebar:
    st.markdown(f"## 🎓 {COLLEGE_NAME}")
    st.caption("AI Help Desk — Admin & Knowledge Base")
 
    if kb_ready:
        st.success("Knowledge base is ready ✅")
    else:
        st.warning("No knowledge base loaded yet")
 
    with st.expander("📊 Knowledge base status", expanded=not kb_ready):
        st.markdown(st.session_state.status_message)
        if st.session_state.doc_registry:
            st.markdown("**Documents indexed:**")
            for entry in st.session_state.doc_registry:
                st.markdown(f"- `[{entry['category']}]` {entry['name']} — {entry['pages']} pg")
 
    st.divider()
    st.markdown("### 🛠️ Admin: Manage Knowledge Base")
 
    if ADMIN_PASSCODE and not st.session_state.admin_unlocked:
        st.info("Enter the admin passcode to manage the knowledge base.")
        entered = st.text_input("Admin passcode", type="password", label_visibility="collapsed",
                                 placeholder="Admin passcode")
        if st.button("Unlock", use_container_width=True):
            if entered == ADMIN_PASSCODE:
                st.session_state.admin_unlocked = True
                st.rerun()
            else:
                st.error("Incorrect passcode.")
    else:
        with st.expander("🚀 Load ready-made demo knowledge base", expanded=not kb_ready):
            st.caption(f"A sample {COLLEGE_NAME} handbook, circulars, and reference docs — "
                       "try the chat before uploading real documents.")
            if st.button("📥 Load Sample Knowledge Base", use_container_width=True):
                load_sample_knowledge_base(append=False)
            if st.button("➕ Add Sample Docs to Existing KB", use_container_width=True):
                load_sample_knowledge_base(append=True)
 
        with st.expander("📤 Upload your own documents"):
            handbook_files = st.file_uploader(
                "📘 Student Handbook", type=["pdf", "txt", "md"],
                accept_multiple_files=True, key="handbook_uploader",
            )
            circular_files = st.file_uploader(
                "📄 Circulars / Notices", type=["pdf", "txt", "md"],
                accept_multiple_files=True, key="circular_uploader",
            )
            other_files = st.file_uploader(
                "🗂️ Other (fees, rules, FAQs, etc.)", type=["pdf", "txt", "md"],
                accept_multiple_files=True, key="other_uploader",
            )
 
            b1, b2 = st.columns(2)
            with b1:
                build_clicked = st.button("🔧 Build KB", type="primary", use_container_width=True)
            with b2:
                append_clicked = st.button("➕ Add to KB", use_container_width=True)
 
            if build_clicked:
                build_knowledge_base(handbook_files, circular_files, other_files, append=False)
            if append_clicked:
                build_knowledge_base(handbook_files, circular_files, other_files, append=True)
 
        with st.expander("💾 Save / load / reset"):
            if st.button("💾 Save current KB to disk", use_container_width=True):
                save_knowledge_base()
            if st.button("📂 Load saved KB from disk", use_container_width=True):
                load_knowledge_base_from_disk()
            if st.button("🗑️ Clear everything", use_container_width=True):
                clear_session()
                st.rerun()
 
 
# ---------------- Main area: student chat only ----------------
st.markdown(
    f"""
    <div class="nexora-header">
        <h1>🎓 {COLLEGE_NAME} — Help Desk</h1>
        <p>Ask about admissions, timetables, fees, or rules. Answers are grounded strictly in the
        official handbook and circulars — nothing is made up.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
if not st.session_state.chat_history:
    st.info(f"👋 Hi! I'm the {COLLEGE_NAME} Help Desk assistant. Ask me anything about "
            "admissions, timetables, fees, or college rules.")
 
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if message["role"] == "assistant" and sources:
            with st.expander("📎 Sources"):
                for line in sources:
                    st.markdown(f"- {line}")
 
# This is the LAST top-level element in the script, with nothing else
# rendered below it at the page level — that's what keeps it pinned to
# the bottom of the screen instead of drifting with the page content.
question = st.chat_input(
    "e.g. 'What is the last date to pay semester fees?'" if kb_ready
    else "Load a knowledge base from the sidebar first…",
    disabled=not kb_ready,
)
 
if question:
    # Show the student's question immediately...
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    # ...then show a spinner while the answer is generated, and only add
    # the assistant's bubble once it's actually ready. No st.rerun() here —
    # that's what stops both messages from popping in at the same instant.
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Checking the handbook and circulars…"):
            answer, sources = get_answer_and_sources(question)
        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources"):
                for line in sources:
                    st.markdown(f"- {line}")
 
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
 