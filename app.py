import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nexora College Helpdesk",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN is missing. Please add it to your .env file.")
    st.stop()


# ============================================================
# 3. CONSTANTS
# ============================================================

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FALLBACK_MESSAGE = (
    "I don't know based on the uploaded college documents."
)

KNOWLEDGE_FOLDER = "knowledge"


# ============================================================
# 4. KNOWLEDGE BASE FILES
# ============================================================

FILES = [
    "college_handbook.pdf",
    "admission.pdf",
    "courses.pdf",
    "fees.pdf",
    "attendance.pdf",
    "examinations.pdf",
    "hostel.pdf",
    "library.pdf",
    "placements.pdf",
    "circulars.pdf",
    "contact_directory.pdf",
    "academic_calendar.pdf",
    "timetable.pdf",
    "scholarships.pdf",
    "student_services.pdf"
]


# ============================================================
# 5. LOAD PDFs AND CREATE VECTOR DATABASE
# ============================================================

@st.cache_resource
def create_vector_database():

    documents = []

    for filename in FILES:

        filepath = os.path.join(
            KNOWLEDGE_FOLDER,
            filename
        )

        if not os.path.exists(filepath):

            st.warning(
                f"Warning: {filepath} was not found."
            )

            continue

        try:

            loader = PyPDFLoader(filepath)

            loaded_documents = loader.load()

            # Store source filename in metadata
            for document in loaded_documents:

                document.metadata["source_file"] = filename

            documents.extend(loaded_documents)

        except Exception as error:

            st.warning(
                f"Could not load {filename}: {error}"
            )


    if not documents:

        st.error(
            "No PDF documents were loaded from the knowledge folder."
        )

        st.stop()


    # --------------------------------------------------------
    # Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


    # --------------------------------------------------------
    # Create FAISS vector database
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embedding
    )


    return vectorstore


# ============================================================
# 6. CREATE VECTOR DATABASE
# ============================================================

with st.spinner("Loading Nexora knowledge base..."):

    vectorstore = create_vector_database()


# ============================================================
# 7. HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    provider="auto",
    token=HF_TOKEN
)


# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are Nexora Institute of Technology's College Helpdesk AI Assistant.

Your job is to answer student questions using ONLY the information
provided in the retrieved college documents.

The college is fictional and all information comes from the college
knowledge base.

IMPORTANT RULES:

1. Use only the retrieved documents.

2. Do not invent information.

3. Do not use outside knowledge.

4. If the answer cannot be found in the documents, say exactly:

"{FALLBACK_MESSAGE}"

5. If the question is ambiguous and the documents contain multiple
possible answers, ask a short clarification question instead of
choosing an answer arbitrarily.

Example:

Student:
"What is the college fee?"

Good response:
"Which program or course are you asking about?"

6. If a follow-up question depends on the previous question, use the
conversation history to understand what the student means.

Example:

Student:
"What are the library timings?"

Assistant:
"The central library is open from 8:00 AM to 8:00 PM on working days."

Student:
"What about during exams?"

Understand that "during exams" refers to the library.

7. DOCUMENT PRIORITY RULE:

If retrieved documents contain both:

- an older/general rule from a handbook, library document, hostel
  document, attendance document, etc.

AND

- a later approved circular that specifically changes or temporarily
  overrides that rule,

use the later applicable circular.

The later circular has priority for that specific situation.

8. Do NOT combine conflicting information from two documents.

Example:

General rule:
Library closes at 8:00 PM.

Later circular:
During examination preparation, library remains open until 10:00 PM.

If the student asks about library timings during the examination
preparation period, answer 10:00 PM.

Do NOT answer 8:00 PM and 10:00 PM together.

9. Pay attention to dates.

If a circular applies only during a specific period, mention that
period when relevant.

10. Keep answers clear and student-friendly.

11. Do not expose private chain-of-thought or hidden reasoning.

12. You may provide a short explanation when useful, but never reveal
hidden internal reasoning.

13. Do not tell the student to refer to the uploaded documents.

14. Do not mention phrases such as:

- "uploaded documents"
- "knowledge base"
- "Sample/Fictional Knowledge Base"
- "refer to the document"

unless the student specifically asks where the information came from.

15. Give the answer directly and concisely.

FEW-SHOT EXAMPLES:

Example 1:

Question:
"What is the minimum attendance?"

Answer:
"The minimum attendance requirement is 75%."

Example 2:

Question:
"Who should I contact for hostel issues?"

Answer:
"You should contact the Hostel Office."

Example 3:

Question:
"What is the college fee?"

Answer:
"Which program or course are you asking about?"

Example 4:

Question:
"What are the library timings during examination preparation?"

Context contains:

- General library timing: until 8:00 PM.
- Later Circular: library remains open until 10:00 PM during
  examination preparation.

Answer:
"During the examination preparation period from 20 June to
5 July 2026, the central library remains open until 10:00 PM
on working days."
"""


# ============================================================
# 9. CREATE SEARCH QUERY
# ============================================================

def create_search_query(question):

    previous_questions = []

    for message in st.session_state.messages:

        if message["role"] == "user":

            previous_questions.append(
                message["content"]
            )


    # Handle follow-up questions

    if previous_questions:

        previous_question = previous_questions[-1]

        return (
            previous_question
            + " "
            + question
        )

    return question


# ============================================================
# 10. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    search_query = create_search_query(question)

    try:

        # Retrieve more documents
        documents = vectorstore.similarity_search(
            search_query,
            k=8
        )


        priority_documents = []

        other_documents = []


        query_lower = search_query.lower()


        # ----------------------------------------------------
        # Check whether question is exam-related
        # ----------------------------------------------------

        exam_related = (
            "exam" in query_lower
            or "examination" in query_lower
            or "exams" in query_lower
        )


        # ----------------------------------------------------
        # Check whether question is library-related
        # ----------------------------------------------------

        library_related = (
            "library" in query_lower
        )


        for document in documents:

            source = document.metadata.get(
                "source_file",
                ""
            ).lower()

            text = document.page_content.lower()


            # Identify circular document

            is_circular = (
                "circular" in source
                or "circular" in text
            )


            # ------------------------------------------------
            # Give circular priority for:
            #
            # Library + Exam questions
            # ------------------------------------------------

            if (
                is_circular
                and exam_related
                and library_related
            ):

                priority_documents.append(
                    document
                )

            else:

                other_documents.append(
                    document
                )


        # Circulars come first

        documents = (
            priority_documents
            + other_documents
        )


        return documents, search_query


    except Exception as error:

        st.error(
            f"Document retrieval error: {error}"
        )

        return [], search_query


# ============================================================
# 11. GET CONVERSATION HISTORY
# ============================================================

def get_conversation_history():

    history = []

    for message in st.session_state.messages:

        if message["role"] in ["user", "assistant"]:

            history.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )


    # Keep only recent conversation

    return history[-6:]


# ============================================================
# 12. CONVERT HF STREAM CHUNKS INTO TEXT
# ============================================================

def generate_response(response_stream):

    for chunk in response_stream:

        # Ignore chunks without choices

        if not chunk.choices:
            continue


        # Get the text from the streaming chunk

        content = chunk.choices[0].delta.content


        # Yield only actual text

        if content:

            yield content


# ============================================================
# 13. ASK AI WITH STREAMING
# ============================================================

def ask_ai(question, context):

    history = get_conversation_history()


    # --------------------------------------------------------
    # Create messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


    # --------------------------------------------------------
    # Add conversation history
    # --------------------------------------------------------

    for message in history:

        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )


    # --------------------------------------------------------
    # Add current question and retrieved context
    # --------------------------------------------------------

    user_prompt = f"""
Retrieved College Documents:

{context}


Student's Current Question:

{question}


Instructions:

Answer the student's question using the retrieved documents.

Remember:

- Use only the documents.
- Do not invent information.
- Follow the document priority rule.
- Later applicable circulars override older general rules.
- If information is missing, say:
  "{FALLBACK_MESSAGE}"
- If the question is ambiguous, ask for clarification.
- Give the answer directly and concisely.
"""


    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )


    try:

        # ====================================================
        # ENABLE STREAMING
        # ====================================================

        response_stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.1,
            stream=True
        )


        # ----------------------------------------------------
        # Convert API chunks into text
        # ----------------------------------------------------

        return generate_response(
            response_stream
        )


    except Exception as error:

        error_message = str(error).lower()


        # ----------------------------------------------------
        # Rate limit error
        # ----------------------------------------------------

        if (
            "rate" in error_message
            or "429" in error_message
        ):

            return (
                "The AI service is temporarily busy. "
                "Please try again in a moment."
            )


        # ----------------------------------------------------
        # Connection error
        # ----------------------------------------------------

        if (
            "timeout" in error_message
            or "connection" in error_message
        ):

            return (
                "I could not connect to the AI service. "
                "Please try again."
            )


        # ----------------------------------------------------
        # Other errors
        # ----------------------------------------------------

        return (
            "Sorry, I encountered an error while "
            "processing your question."
        )


# ============================================================
# 14. GET SOURCE DOCUMENTS
# ============================================================

def get_sources(documents):

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source_file",
            "Unknown document"
        )


        if source not in sources:

            sources.append(source)


    return sources


# ============================================================
# 15. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 16. HEADER
# ============================================================

st.title("🎓 Nexora College Helpdesk")

st.write(
    "Ask questions about admissions, courses, fees, hostel, "
    "library, examinations, placements, scholarships, "
    "attendance and student services."
)


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 Nexora Helpdesk")

    st.write(
        "RAG-Based AI Assistant"
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write(
        f"{len(FILES)} college documents loaded"
    )

    st.divider()

    st.subheader("Example Questions")

    st.write(
        "• What is the minimum attendance?"
    )

    st.write(
        "• What are the library timings?"
    )

    st.write(
        "• What are the library timings during exams?"
    )

    st.write(
        "• What is the hostel fee?"
    )

    st.write(
        "• Who should I contact for hostel issues?"
    )

    st.write(
        "• What are the admission requirements?"
    )

    st.write(
        "• What scholarships are available?"
    )

    st.write(
        "• What are the placement services?"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 18. DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    role = message["role"]


    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    if role == "user":

        with st.chat_message("user"):

            st.write(
                message["content"]
            )


    # --------------------------------------------------------
    # Assistant message
    # --------------------------------------------------------

    elif role == "assistant":

        with st.chat_message("assistant"):

            st.write(
                message["content"]
            )


            # Display sources

            if message.get("sources"):

                with st.expander(
                    "📚 Sources used"
                ):

                    for source in message["sources"]:

                        st.write(
                            f"• {source}"
                        )


# ============================================================
# 19. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask Nexora College Helpdesk..."
)


# ============================================================
# 20. PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(question)


    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    with st.spinner(
        "Searching college documents..."
    ):

        documents, search_query = retrieve_documents(
            question
        )


    # ========================================================
    # NO DOCUMENTS FOUND
    # ========================================================

    if not documents:

        answer = FALLBACK_MESSAGE

        sources = []


        with st.chat_message("assistant"):

            st.write(answer)


    # ========================================================
    # DOCUMENTS FOUND
    # ========================================================

    else:

        # ----------------------------------------------------
        # Create context
        # ----------------------------------------------------

        context_parts = []


        for document in documents:

            source = document.metadata.get(
                "source_file",
                "Unknown document"
            )

            content = document.page_content


            context_parts.append(
                f"""
SOURCE: {source}

CONTENT:
{content}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # Generate streaming response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Preparing answer..."
            ):

                response_stream = ask_ai(
                    question,
                    context
                )


            # ------------------------------------------------
            # If an error occurred before streaming
            # ------------------------------------------------

            if isinstance(
                response_stream,
                str
            ):

                answer = response_stream

                st.write(answer)


            else:

                # --------------------------------------------
                # STREAM ONLY THE TEXT
                # --------------------------------------------

                answer = st.write_stream(
                    response_stream
                )


        # ----------------------------------------------------
        # Get source names
        # ----------------------------------------------------

        sources = get_sources(
            documents
        )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            with st.chat_message("assistant"):

                with st.expander(
                    "📚 Sources used"
                ):

                    for source in sources:

                        st.write(
                            f"• {source}"
                        )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )