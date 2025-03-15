from webscrap import scrape_and_clean
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()



# ---------------------------- Step 1: Setup Pinecone --------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ragchat"

# pc.create_index(
#         name=index_name,
#         dimension=1536,  # OpenAI embedding dimension
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

index = pc.Index(index_name)


# ------------------------- Step 2: Scrape and Chunk Data ----------------------------

# Step 1: Scrape webpage content
url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
full_text = scrape_and_clean(url)

# Step 2: Chunk the text
text_splitter = CharacterTextSplitter(
    separator="",        # Split at every character if needed
    chunk_size=250,       
    chunk_overlap=10,     
    length_function=len
)
chunks = text_splitter.split_text(full_text)
# for i in range(len(chunks)):
#     print(f"Chunk {i}: {chunks[i]}")



# ------------------------- Step 3: Store Chunks as Embeddings ----------------------------

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
for i, chunk in enumerate(chunks):
    chunk_embedding = embeddings.embed_query(chunk)
    index.upsert([(str(i), chunk_embedding, {"text": chunk})])

print(f"Successfully added {len(chunks)} chunks to Pinecone!")


# ------------------------- Step 4: LLM Chatbot Creation ----------------------------

llm = ChatOpenAI(model="gpt-4", temperature=0)

#User Asks Question
while True:
    query = input("Ask a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    #  Retrieve top 3 relevant chunks from Pinecone

    query_embedding = embeddings.embed_query(query)
    result = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    # Combine top results into a single context string
    augmented_context = "\n\n".join([match.metadata["text"] for match in result.matches])


    # Create Prompt Template

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the context provided to answer the question accurately. Only use this context, not your knowledge.\n\n"
                "Context:{context}"
                "Question:{question}"
                
    )

    # Set up LangChain's LLMChain with GPT-4 and prompt template
    chain = prompt | llm 

    # Generate GPT-4 response using augmented context and user question
    response = chain.invoke({"context": augmented_context, "question": query})
    # print(f"Retrieved Text: {augmented_context}")
    print("Answer:", response.content)