import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph , START , END 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 

load_dotenv()

#Step 1 - Building the RAG retrievers 
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )

def build_retriever(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
    chunks = splitter.split_documents(document)
    vectorstore = FAISS.from_documents(chunks,embeddings)
    return vectorstore.as_retriever(search_kwargs = {"k":4})

academic_retriever = build_retriever("ThakurAcademic.pdf")
fee_retriever = build_retriever("thakurfees.pdf")
placement_retriever = build_retriever("thakurplacement.pdf") # Add this PDF

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

#step2 - State 
class State(TypedDict):
    programme : str  
    messages : Annotated[list,add_messages]
    query_type : str 
    retrieved_context : str 

#Step 3 - Nodes generation 
    
def classifier_node(state : State) -> dict:
    """Refined classifier to separate academic, fee, placement, general"""
    last_message = state['messages'][-1].content

    prompt = f"""
    Classify the following student query into exactly one category: 
    "academic", "fee", "placement" or "general".

    Rules:
    Use "academic" for: attendance, exams, grading, credits, promotion, ATKT, 
    course structure, syllabus, internship hours, degree requirements, semester subjects.
    Examples: "what is minimum attendance", "how many credits in Sem 5 CSE"

    Use "fee" for: tuition, payment, late fee charges, refund, ebc scholarship, 
    tfws, st/sc fee, female fee, installment, dues, scholarship.
    Examples: "what is B.Tech fee", "ebc scholarship details"

    Use "placement" for: packages, placement rate, highest package, lowest package, 
    average package, companies visiting, internships with stipend, placement eligibility.
    Examples: "what is highest placement", "which companies came", "average package for IT"

    Use "general" for: greetings, casual talk, or anything not related to college rules, 
    fee or placement.

    Query: {last_message}

    Return only one word: academic, fee, placement, or general.
    """

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    elif "placement" in category:
        category = "placement"
    else:
        category = "general"
    
    return {"query_type" : category}

def academic_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def placement_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = placement_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def general_node(state: State) -> dict:
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state: State) -> dict:
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly TCET college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a TCET college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}

#step 4 - router function 
def route_query(state:State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    elif state['query_type'] == "placement":
        return "placement_rag"
    else:
        return "general"

#step 5 - Building the graph 
graph = StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("placement_rag",placement_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

#edges 
graph.add_edge(START,"classifier")
graph.add_conditional_edges("classifier",route_query)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("placement_rag","response")
graph.add_edge("general","response")
graph.add_edge("response",END)

app = graph.compile()


programmes = [
    "1. B.E. Computer Engineering",
    "2. B.E. Information Technology", 
    "3. B.E. Electronics & Telecommunication Engg.",
    "4. B.E. Electronics Engineering",
    "5. B.E. Mechanical Engineering",
    "6. B.E. Civil Engineering",
    "7. B.E. Computer Science & Engineering",
    "8. B.E. Mechanical & Mechatronics Engg.",
    "9. B.E. Electronics & Computer Science",
    "10. B.Tech. Artificial Intelligence & Data Science",
    "11. B.Tech. Artificial Intelligence & Machine Learning",
    "12. B.Tech. Internet of Things",
    "13. B.Voc. AI&DS",
    "14. B.Voc. Software Development",
    "15. MCA",
    "16. M.E."
]

programme_map = {
    "1": "B.E. Computer Engineering",
    "2": "B.E. Information Technology", 
    "3": "B.E. Electronics & Telecommunication Engineering",
    "4": "B.E. Electronics Engineering",
    "5": "B.E. Mechanical Engineering",
    "6": "B.E. Civil Engineering",
    "7": "B.E. Computer Science & Engineering",
    "8": "B.E. Mechanical & Mechatronics Engineering",
    "9": "B.E. Electronics & Computer Science",
    "10": "B.Tech. Artificial Intelligence & Data Science",
    "11": "B.Tech. Artificial Intelligence & Machine Learning",
    "12": "B.Tech. Internet of Things",
    "13": "B.Voc. Artificial Intelligence & Data Science",
    "14": "B.Voc. Software Development",
    "15": "MCA",
    "16": "M.E."
}