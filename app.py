import os
import re
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

KNOWLEDGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")


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

16. Answer ONLY the student's current question.

17. Use only information that directly answers the current question.
Do not include unrelated information merely because it appears in a
retrieved document.

18. Do not summarize an entire document when the student asks for one
specific fact.

19. For admission questions, use only admission requirements and
admission-process information. Do NOT include attendance rules,
examination rules, course lists, hostel information, or other handbook
content unless the student specifically asks for it.

20. For fee questions, use only fee-related information. If the student
asks for "all fees" or "fee structure", list the fee names and amounts
from the fee document. Do not add payment schedules, refund policies,
attendance rules, admission rules, or other unrelated information.

21. Preserve amounts and currency symbols exactly as stated in the
retrieved college document. Never convert currency or invent a currency
symbol.

22. Do not describe the fees as "sample" or "fictional" unless the
student specifically asks about that.

23. Do not tell the student to refer to a document or circular unless
the student specifically asks for the source.

24. If the student asks only "fee", "fees", "all fee", "all fees", or
"fee structure", answer with the fee list from fees.pdf. Do not say
that you do not know, and do not ask which fee unless the question
clearly refers to a specific program or fee type.

25. Never use the words "sample", "fictional", or "subject to change"
in an answer unless the student explicitly asks whether the college
data is sample or fictional.

26. For attendance questions, answer the attendance requirement directly.
Do not add the phrase "standard sample policy" or unrelated examination
rules unless specifically asked.

27. Never convert or remove the currency symbol from a monetary amount.
Use the exact currency symbol and amount shown in the retrieved document.

28. For fee questions, reproduce the fee names and amounts from
fees.pdf accurately. Do not add commentary about the data being sample,
fictional, subject to change, or needing to be checked elsewhere.

29. For short valid topic questions such as "fee", "fees", "attendance",
or "library", answer from the matching college document rather than
treating the question as unsupported.

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

def normalize_question(question):
    """Normalize common student typos without changing the meaning."""
    q = question.strip()
    replacements = {
        "timimngs": "timings",
        "timngs": "timings",
        "timingss": "timings",
        "admisson": "admission",
        "admisison": "admission",
        "scholorship": "scholarship",
        "attendence": "attendance",
        "examinaton": "examination",
        "helpdest": "helpdesk",
        "helpdeskk": "helpdesk",
        "help desk": "helpdesk",
        "student help dest": "student helpdesk",
    }
    for wrong, right in replacements.items():
        q = re.sub(r"\b" + re.escape(wrong) + r"\b", right, q, flags=re.IGNORECASE)
    return q


def get_previous_user_question():
    """Return the last user question before the current one."""
    users = [m["content"].strip() for m in st.session_state.messages if m["role"] == "user"]
    if len(users) >= 2:
        return users[-2]
    return None


def create_search_query(question):
    current_question = normalize_question(question)
    current_lower = current_question.lower()
    previous_question = get_previous_user_question()

    # Resolve short/contextual follow-ups BEFORE checking explicit topics.
    # Example: after "What are the library timings?", "timimngs during exams"
    # must be interpreted as a library question, not as a generic exam query.
    follow_up_phrases = [
        "what about", "how about", "and during", "during",
        "what is the amount", "how much", "how many", "when",
        "where", "who should", "who do", "how do i", "can i",
        "is it", "does it"
    ]
    short_follow_up = len(current_question.split()) <= 6
    previous_lower = previous_question.lower() if previous_question else ""
    previous_topic = any(topic in previous_lower for topic in [
        "library", "hostel", "fee", "fees", "admission", "scholarship",
        "placement", "attendance", "course", "timetable", "exam",
        "examination", "student service"
    ])

    if previous_question and (any(p in current_lower for p in follow_up_phrases) or (short_follow_up and previous_topic)):
        # A clearly named new object starts a new question.
        explicit_new_object = any(topic in current_lower for topic in [
            "fee", "fees", "admission", "admissions", "course", "courses",
            "scholarship", "scholarships", "placement", "placements", "hostel",
            "attendance", "library", "timetable", "student service"
        ])

        # Special case: "timings during exams" after a library question.
        if (short_follow_up and "library" in previous_lower and
                ("exam" in current_lower or "timing" in current_lower) and
                "library" not in current_lower):
            return normalize_question(previous_question) + " " + current_question

        if not explicit_new_object:
            return normalize_question(previous_question) + " " + current_question

    # Normalize very short fee questions so they always retrieve the complete
    # fee document instead of relying on semantic similarity.
    fee_question = current_lower.strip(" ?.!,-")
    if fee_question in {"fee", "fees", "all fee", "all fees", "fee structure", "fees structure"}:
        return (
            "complete fee structure tuition fee university academic charges "
            "laboratory technology fee library student services fee "
            "examination fee hostel accommodation hostel mess advance"
        )

    explicit_topics = [
        "fee", "fees", "tuition", "cost", "charge", "charges",
        "admission", "admissions", "eligibility", "apply",
        "course", "courses", "branch", "branches", "program", "programs", "degree",
        "scholarship", "scholarships", "placement", "placements", "recruitment", "career", "job",
        "hostel", "warden", "mess", "accommodation", "attendance", "absent", "absence",
        "library", "librarian", "exam", "examination", "exams", "hall ticket",
        "timetable", "schedule", "academic calendar", "semester date", "academic year",
        "contact", "phone", "email", "number", "office", "student service", "student services",
        "counselling", "transport", "id card"
    ]
    if any(topic in current_lower for topic in explicit_topics):
        return current_question

    return current_question


# ============================================================
# 10. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    current_question = question.strip()
    question_lower = current_question.lower()

    try:

        # ----------------------------------------------------
        # FEE QUESTIONS — HARD ROUTE TO fees.pdf CHUNKS
        # ----------------------------------------------------
        # This check happens BEFORE search-query creation.
        # Therefore "fee", "fees", "all fee", "all fees",
        # "fee structure", etc. can never fail because of
        # semantic similarity.
        # ----------------------------------------------------

        fee_words = [
            "fee",
            "fees",
            "tuition",
            "cost",
            "charge",
            "charges"
        ]

        is_fee_question = any(
            word in question_lower
            for word in fee_words
        )

        if is_fee_question:

            # Load fees.pdf directly for every fee question.
            # This completely bypasses FAISS similarity and metadata
            # filtering, so "fee", "fees", "all fee", and "all fees"
            # are handled deterministically.
            fee_path = os.path.join(
                KNOWLEDGE_FOLDER,
                "fees.pdf"
            )

            if not os.path.isfile(fee_path):

                st.error(
                    f"Fee document not found: {os.path.abspath(fee_path)}"
                )

                return [], current_question

            try:

                fee_documents = PyPDFLoader(
                    fee_path
                ).load()

                for document in fee_documents:
                    document.metadata["source_file"] = "fees.pdf"

                if fee_documents:
                    return fee_documents, current_question

                st.error(
                    "fees.pdf was loaded but contains no readable text."
                )

                return [], current_question

            except Exception as error:

                st.error(
                    f"Could not read fees.pdf: {error}"
                )

                return [], current_question


        # ----------------------------------------------------
        # CREATE SEARCH QUERY FOR NON-FEE QUESTIONS
        # ----------------------------------------------------

        search_query = create_search_query(
            current_question
        )

        query_lower = search_query.lower()


        # ----------------------------------------------------
        # LIBRARY + EXAMINATION SPECIAL CASE
        # ----------------------------------------------------

        library_related = (
            "library" in query_lower
        )

        exam_related = (
            "exam" in query_lower
            or "examination" in query_lower
            or "exams" in query_lower
        )

        if library_related and exam_related:

            circular_documents = []

            for document in vectorstore.docstore._dict.values():

                source = document.metadata.get(
                    "source_file",
                    ""
                ).lower()

                content = document.page_content.lower()

                if (
                    source == "circulars.pdf"
                    and (
                        "library timing update" in content
                        or "10:00 pm" in content
                    )
                ):

                    circular_documents.append(
                        document
                    )

            if circular_documents:

                return circular_documents, search_query


        # ----------------------------------------------------
        # DOCUMENT CATEGORY
        # ----------------------------------------------------

        source_files = None


        if (
            "hostel" in query_lower
            or "warden" in query_lower
            or "mess" in query_lower
            or "accommodation" in query_lower
        ):

            source_files = [
                "hostel.pdf",
                "contact_directory.pdf",
                "circulars.pdf"
            ]


        elif (
            "library" in query_lower
            or "librarian" in query_lower
        ):

            source_files = [
                "library.pdf"
            ]


        elif (
            "admission" in query_lower
            or "admissions" in query_lower
            or "eligibility" in query_lower
            or "apply" in query_lower
        ):

            source_files = [
                "admission.pdf"
            ]


        elif (
            "scholarship" in query_lower
            or "scholarships" in query_lower
        ):

            source_files = [
                "scholarships.pdf",
                "student_services.pdf"
            ]


        elif (
            "placement" in query_lower
            or "placements" in query_lower
            or "recruitment" in query_lower
            or "career" in query_lower
            or "job" in query_lower
        ):

            source_files = [
                "placements.pdf",
                "circulars.pdf"
            ]


        elif (
            "attendance" in query_lower
            or "absent" in query_lower
            or "absence" in query_lower
        ):

            source_files = [
                "attendance.pdf",
                "circulars.pdf"
            ]


        elif (
            "exam" in query_lower
            or "examination" in query_lower
            or "hall ticket" in query_lower
        ):

            source_files = [
                "examinations.pdf",
                "circulars.pdf"
            ]


        elif (
            "course" in query_lower
            or "courses" in query_lower
            or "branch" in query_lower
            or "branches" in query_lower
            or "program" in query_lower
            or "programs" in query_lower
            or "degree" in query_lower
        ):

            source_files = [
                "courses.pdf",
                "college_handbook.pdf"
            ]


        elif (
            "contact" in query_lower
            or "phone" in query_lower
            or "email" in query_lower
            or "number" in query_lower
            or "office" in query_lower
        ):

            source_files = [
                "contact_directory.pdf",
                "student_services.pdf"
            ]


        elif (
            "timetable" in query_lower
            or "schedule" in query_lower
            or "class timing" in query_lower
        ):

            source_files = [
                "timetable.pdf",
                "academic_calendar.pdf"
            ]


        elif (
            "academic calendar" in query_lower
            or "semester date" in query_lower
            or "academic year" in query_lower
        ):

            source_files = [
                "academic_calendar.pdf",
                "circulars.pdf"
            ]


        elif (
            "student service" in query_lower
            or "student services" in query_lower
            or "counselling" in query_lower
            or "transport" in query_lower
            or "id card" in query_lower
        ):

            source_files = [
                "student_services.pdf",
                "contact_directory.pdf"
            ]


        # ----------------------------------------------------
        # CATEGORY RETRIEVAL
        # ----------------------------------------------------

        if source_files:

            candidates = vectorstore.similarity_search(
                search_query,
                k=30
            )

            filtered_documents = []

            for document in candidates:

                source = document.metadata.get(
                    "source_file",
                    ""
                )

                if source in source_files:

                    filtered_documents.append(
                        document
                    )

            # Guaranteed category fallback from the already
            # loaded vectorstore.
            if not filtered_documents:

                for document in vectorstore.docstore._dict.values():

                    source = document.metadata.get(
                        "source_file",
                        ""
                    )

                    if source in source_files:

                        filtered_documents.append(
                            document
                        )

            return filtered_documents[:12], search_query


        # ----------------------------------------------------
        # GENERAL QUESTIONS
        # ----------------------------------------------------

        scored_documents = (
            vectorstore
            .similarity_search_with_relevance_scores(
                search_query,
                k=15
            )
        )

        MIN_RELEVANCE_SCORE = 0.15

        relevant_documents = [
            document
            for document, score in scored_documents
            if score >= MIN_RELEVANCE_SCORE
        ]

        return relevant_documents[:8], search_query


    except Exception as error:

        st.error(
            f"Document retrieval error: {error}"
        )

        return [], current_question


# ============================================================
# 11. CONVERSATION HISTORY
# ============================================================

# Previous generated answers are intentionally not sent back to the model.
# Follow-up context is handled only by retrieval.


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

    user_prompt = f"""
RETRIEVED COLLEGE DOCUMENTS:

{context}

CURRENT STUDENT QUESTION:

{question}

STRICT INSTRUCTIONS:
- Answer ONLY the current question.
- Use ONLY facts explicitly present in the retrieved documents above.
- Do not use previous answers or outside knowledge.
- Do not invent dates, amounts, deadlines, policies, schedules,
  conditions, contacts, or explanations.
- If the requested information is not explicitly present, answer exactly:
  "{FALLBACK_MESSAGE}"
- If the question asks for fees, list only the fee information relevant
  to the question. Do not add payment/refund information unless asked.
- Preserve every amount, date, percentage, and time exactly as stated.
- For library questions during examination preparation, use the later
  applicable circular instead of the older general library timing.
- Keep the answer concise.
"""

    try:
        response_stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=250,
            temperature=0.0,
            stream=True
        )

        for chunk in response_stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as error:
        error_message = str(error).lower()
        if "rate" in error_message or "429" in error_message:
            yield "The AI service is temporarily busy. Please try again in a moment."
        elif "timeout" in error_message or "connection" in error_message:
            yield "I could not connect to the AI service. Please try again."
        else:
            yield "Sorry, I encountered an error while processing your question."


# ============================================================
# 14. EXACT FEE LIST FOR A GENERAL FEES QUESTION
# ============================================================

def exact_fee_list(question):
    q = question.lower().strip(" ?.!,-")
    if q not in {"fee", "fees", "all fee", "all fees", "fee structure", "fees structure"}:
        return None

    fee_path = os.path.join(KNOWLEDGE_FOLDER, "fees.pdf")
    try:
        docs = PyPDFLoader(fee_path).load()
    except Exception:
        return None

    text = "\n".join(doc.page_content for doc in docs)
    wanted = [
        "B.Tech Tuition Fee",
        "University/Academic Charges",
        "Laboratory and Technology Fee",
        "Library and Student Services Fee",
        "Examination Fee",
        "Hostel Accommodation",
        "Hostel Mess Advance"
    ]

    lines = []
    for name in wanted:
        pattern = re.compile(
            re.escape(name) + r"\s*[\n: -]*[■₹$]?\s*([0-9][0-9,]*)\s*(per academic year|per semester|per year)",
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            lines.append(f"- {name}: ₹{match.group(1)} {match.group(2)}")

    if not lines:
        return None

    return "The fees for Nexora Institute of Technology include:\n\n" + "\n".join(lines)


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


def exact_common_answer(question):
    """Deterministic answers for high-confidence, frequently asked facts."""
    q = normalize_question(question).lower().strip(" ?.!,-")
    search_q = create_search_query(question).lower()

    # Library timings — general rule.
    if "library" in q and any(x in q for x in ["timing", "timings", "hours", "open", "closing", "close"]):
        if "exam" in q or "examination" in q or "exam" in search_q or "examination" in search_q:
            return (
                "During the examination preparation period from 20 June to 5 July 2026, "
                "the central library will remain open until 10:00 PM on working days."
            ), "circulars.pdf"
        return (
            "The central library is open from 8:00 AM to 8:00 PM Monday through Friday "
            "and from 9:00 AM to 1:00 PM on Saturdays. The library remains closed on "
            "Sundays and declared institute holidays unless a special notice states otherwise."
        ), "library.pdf"

    # A short follow-up such as "timings during exams" after a library question.
    if ("library" in search_q and ("exam" in search_q or "examination" in search_q)
            and any(x in q for x in ["timing", "timings", "hours", "open", "close", "closing", "during"])):
        return (
            "During the examination preparation period from 20 June to 5 July 2026, "
            "the central library will remain open until 10:00 PM on working days."
        ), "circulars.pdf"

    # Attendance — keep a short direct answer for the standard question.
    if q in {"attendance", "minimum attendance", "what is the minimum attendance", "what is the attendance requirement"}:
        return "The minimum attendance requirement at Nexora Institute of Technology is 75 percent.", "attendance.pdf"

    # Hostel accommodation fee — deterministic so the LLM cannot change
    # the currency or amount (for example, to an incorrect currency).
    hostel_fee_phrases = {
        "hostel fee",
        "hostel fees",
        "what is the hostel fee",
        "what are the hostel fees",
        "how much is hostel",
        "how much is the hostel fee",
        "hostel accommodation fee",
        "what is the hostel accommodation fee",
        "hostel accommodation fees"
    }
    if q in hostel_fee_phrases or ("hostel" in q and "fee" in q):
        return "The hostel accommodation fee at Nexora Institute of Technology is ₹60,000 per year.", "fees.pdf"

    # Contact directory — deterministic answers for named offices so
    # contact information is never lost to semantic retrieval or LLM
    # generation. These values come from contact_directory.pdf.
    contacts = {
        "student helpdesk": ("Student Helpdesk", "+91 80000 10008", "helpdesk@nexora.edu"),
        "admissions": ("Admissions", "+91 80000 10001", "admissions@nexora.edu"),
        "admission": ("Admissions", "+91 80000 10001", "admissions@nexora.edu"),
        "accounts": ("Accounts", "+91 80000 10002", "accounts@nexora.edu"),
        "examinations": ("Examinations", "+91 80000 10003", "exams@nexora.edu"),
        "examination": ("Examinations", "+91 80000 10003", "exams@nexora.edu"),
        "exams": ("Examinations", "+91 80000 10003", "exams@nexora.edu"),
        "training placement": ("Training & Placement", "+91 80000 10004", "placements@nexora.edu"),
        "placement": ("Training & Placement", "+91 80000 10004", "placements@nexora.edu"),
        "placements": ("Training & Placement", "+91 80000 10004", "placements@nexora.edu"),
        "hostel": ("Hostel", "+91 80000 10005", "hostel@nexora.edu"),
        "library": ("Library", "+91 80000 10006", "library@nexora.edu"),
        "academic office": ("Academic Office", "+91 80000 10007", "academics@nexora.edu"),
        "academics": ("Academic Office", "+91 80000 10007", "academics@nexora.edu"),
        "it support": ("IT Support", "+91 80000 10009", "itsupport@nexora.edu"),
        "it": ("IT Support", "+91 80000 10009", "itsupport@nexora.edu"),
        "principal": ("Principal", "+91 80000 10010", "principal@nexora.edu")
    }

    contact_words = ("contact", "phone", "number", "email", "helpdesk", "office")
    if any(word in q for word in contact_words):
        # Return the complete directory when the user asks generally.
        if q in {"contact", "contacts", "contact details", "contact information", "all contacts", "contact numbers"}:
            rows = [
                contacts["student helpdesk"], contacts["admissions"], contacts["accounts"],
                contacts["examinations"], contacts["training placement"], contacts["hostel"],
                contacts["library"], contacts["academic office"], contacts["it support"], contacts["principal"]
            ]
            answer = "Available contact details:\n\n" + "\n".join(
                f"- {name}: {phone} | {email}" for name, phone, email in rows
            )
            return answer, "contact_directory.pdf"

        # Match the most specific office first.
        aliases = [
            ("student helpdesk", "student helpdesk"),
            ("helpdesk", "student helpdesk"),
            ("admissions", "admissions"),
            ("admission", "admission"),
            ("accounts", "accounts"),
            ("examination", "examinations"),
            ("exams", "exams"),
            ("training", "training placement"),
            ("placement", "placement"),
            ("placements", "placements"),
            ("hostel", "hostel"),
            ("library", "library"),
            ("academic office", "academic office"),
            ("academics", "academics"),
            ("it support", "it support"),
            ("principal", "principal")
        ]
        for phrase, key in aliases:
            if phrase in q:
                name, phone, email = contacts[key]
                return f"**{name} contact:** {phone}\n**Email:** {email}", "contact_directory.pdf"

    return None, None


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
    # Deterministic high-confidence answers first
    # --------------------------------------------------------

    common_answer, common_source = exact_common_answer(question)
    exact_answer = exact_fee_list(question)

    if common_answer is not None:

        answer = common_answer
        sources = [common_source]

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("📚 Sources used"):
                st.write(f"• {common_source}")

    elif exact_answer is not None:

        answer = exact_answer
        sources = ["fees.pdf"]

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("📚 Sources used"):
                st.write("• fees.pdf")

    else:
        # Retrieve only when the question is not a deterministic FAQ.
        with st.spinner("Searching college documents..."):
            documents, search_query = retrieve_documents(question)

        if not documents:
            answer = FALLBACK_MESSAGE
            sources = []

            with st.chat_message("assistant"):
                st.write(answer)

        else:

            context_parts = []

            for document in documents:
                source = document.metadata.get("source_file", "Unknown document")
                content = document.page_content
                context_parts.append(f"SOURCE: {source}\n\nCONTENT:\n{content}")

            context = "\n\n".join(context_parts)
            sources = get_sources(documents)

            with st.chat_message("assistant"):
                with st.spinner("Preparing answer..."):
                    answer = st.write_stream(ask_ai(question, context))

                if not answer:
                    answer = FALLBACK_MESSAGE

                if sources:
                    with st.expander("📚 Sources used"):
                        for source in sources:
                            st.write(f"• {source}")


    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )