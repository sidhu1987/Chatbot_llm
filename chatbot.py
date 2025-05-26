import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ Import for local embeddings

# 🔐 Your OpenAI API Key
#OPENAI_API_KEY = "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"
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key='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',
    temperature=0,

)
# Main title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🤖 Q&A Chatbot</h1>", unsafe_allow_html=True)

# Sidebar customization
with st.sidebar:
    st.markdown("<h2 style='color: #306998;'>📄 U R Docs</h2>", unsafe_allow_html=True)
    st.markdown("Upload any PDF file and ask questions based on its content.")
    file = st.file_uploader("📁 Upload Your PDF File", type="pdf")

# Main logic
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # ✅ Generate local embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user input
    user_question = st.text_input("💬 Enter your question about the PDF")

    if user_question:
        # Search similar chunks
        match = vector_store.similarity_search(user_question)


        # Load QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run QA
        response = chain.run(input_documents=match, question=user_question)

        # Show result
        st.markdown("### 🤖 Answer:")
        st.write(response)
