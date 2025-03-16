from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# -------------------- ENV SETUP ----------------------
load_dotenv()

# -------------------- FASTAPI INIT ----------------------
app = FastAPI(title="AI Chatbot Backend", description="Retrieve answers using RAG pipeline", version="1.0")

# -------------------- PINECONE & LLM SETUP ----------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ragchatbot")  # assuming this is your index name
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4", temperature=0.2)  # default temp, can be parameterized

# -------------------- REQUEST MODEL ----------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3  # Default top_k = 3, can be overridden by user


# -------------------- API ENDPOINT ----------------------
@app.post("/query")
def query_chatbot(request: QueryRequest):
    try:
        # Step 1: Embed user query
        query_embedding = embeddings.embed_query(request.question)

        # Step 2: Query Pinecone
        result = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )

        if not result.matches:
            raise HTTPException(status_code=404, detail="No relevant chunks found for this query.")

        # Step 3: Prepare context from retrieved chunks
        context = "\n\n".join([match.metadata["text"] for match in result.matches])

        # Step 4: Create Prompt Template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Use ONLY the provided context to answer the question. "
                "Do NOT use external knowledge.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )

        # Step 5: Call LLM with constructed prompt
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": request.question})

        return {
            "question": request.question,
            "answer": response.content,
            "context_used": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
