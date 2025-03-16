import streamlit as st
from webscrap import scrape_and_clean
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# -------------------------- ENV SETUP --------------------------
load_dotenv()

# -------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="⚙️ Chat with HTML Page - Powered by DataSense", 
                   page_icon="🧠", 
                   layout="wide")

st.title("⚙️ Chat with HTML Page - Powered by DataSense")
st.caption("💬 Ask anything about the scraped webpage, and get intelligent answers. Built by DataSense 🚀")

# -------------------------- SIDEBAR ------------------------------
with st.sidebar:
    st.header("🔧 Control Panel")
    show_chunks = st.toggle("🔍 Show Text Chunks", value=False)
    show_context = st.toggle("📜 Show Context Used for Answer", value=False)
    top_k = st.slider("📈 Number of Chunks to Fetch (Top-K)", 1, 5, 3)
    temperature = st.slider("🔥 Model Creativity (Temperature)", 0.0, 1.0, 0.2)

st.markdown("---")

# -------------------------- PINECONE SETUP -------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ragchatbot"
index = pc.Index(index_name)

# -------------------------- SCRAPE AND CHUNK -------------------------

@st.cache_data(show_spinner="🔎 Scraping and preparing data...")
def load_and_chunk_data(url):
    # Step 1: Scrape webpage content
    full_text = scrape_and_clean(url)

    # Step 2: Chunk the text
    text_splitter = CharacterTextSplitter(
        separator="",        # Split at every character if needed
        chunk_size=250,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    return full_text, chunks

url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
full_text, chunks = load_and_chunk_data(url)

if show_chunks:
    st.subheader("📚 Scraped and Chunked Text Data")
    for idx, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {idx+1}:** {chunk}")

# -------------------------- EMBEDDINGS -------------------------------
@st.cache_resource(show_spinner="🔑 Preparing embeddings...")
def prepare_embeddings(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    for i, chunk in enumerate(chunks):
        chunk_embedding = embeddings.embed_query(chunk)
        index.upsert([(str(i), chunk_embedding, {"text": chunk})])
    return embeddings

embeddings = prepare_embeddings(chunks)
llm = ChatOpenAI(model="gpt-4", temperature=temperature)

# ------------------------- CHAT FUNCTION ---------------------------
def chat_with_html(query, top_k):
    # Embed user query
    query_embedding = embeddings.embed_query(query)
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Combine top results into a single context string
    augmented_context = "\n\n".join([match.metadata["text"] for match in result.matches])

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the context provided to answer the question accurately. "
                 "Only use this context, not your knowledge.\n\n"
                 "Context:\n{context}\n\n"
                 "Question: {question}"
    )

    # Build chain and invoke
    chain = prompt | llm
    response = chain.invoke({"context": augmented_context, "question": query})
    return response.content, augmented_context

# -------------------------- CHATBOT UI -------------------------------
st.subheader("💬 Ask Your Question")
query = st.text_input("Type your question below 👇", placeholder="E.g., What is this book about?")

if st.button("🚀 Get Answer"):
    if query.strip() == "":
        st.error("❌ Please type a valid question!")
    else:
        with st.spinner("🤖 Thinking..."):
            answer, context_used = chat_with_html(query, top_k)
        st.success("✅ Answer Generated:")
        st.markdown(f"**💡 Answer:** {answer}")

        if show_context:
            st.subheader("📜 Context Used to Generate Answer")
            st.write(context_used)

# -------------------------- FOOTER -------------------------------
st.markdown("---")
st.markdown("🎓 **Built by DataSense | AI-Powered Web Scraping & Chatbot System**")
