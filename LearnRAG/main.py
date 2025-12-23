import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def initialize_components():
    #  Intialize the embeddings Model
    embeddings = OpenAIEmbeddings()

    #  Initialize the LLM Model
    llm = ChatOpenAI()

    #  Initialize the Vector Store
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    )

    #  Initialize the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt_template = ChatPromptTemplate.from_template(
        """
           Answer the question based only on the following context:
           {context}

            Question: {question}
            Provide a detailed anser:
        """
    )
    return embeddings, llm, retriever, prompt_template

def format_docs(docs):
    """Formats a list of documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def retrieval_chain_without_lecel(query: str):
    """
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    -No built-in streaming support
    - No assync support without additional code
    - Harder to compose withother chains
    - More verbose and error-prone
    """
    #  Step1: Retrieve relevant documents
    docs = retriever.invoke(query)

    #  Step2: Format the retrieved documents
    context = format_docs(docs)

    #  Step3: Format the prompt with context and question
    prompt = prompt_template.format_messages(context=context, question=query)

    #  Step4: Generate the response using the LLM
    response = llm.invoke(prompt)
    return response.content
    
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LangChain Expression Language (LCEL).
    Returns a chain that can be invoked with {"question: : "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain
    

if __name__ == "__main__":
    print("Initializing Components...")
    embeddings, llm, retriever, prompt_template = initialize_components()
    print("Components Initialized.")

    query = "What is pinecone in machine learning?"

    #  Option 0: Raw invocation without RAG
    print("\n--- Retrieval Chain without RAG ---")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(f"Response: {result_raw.content}")

    #  Option 1: Use implementation without LCEL
    print("\n--- Retrieval Chain without LCEL ---")
    result_without_lcel = retrieval_chain_without_lecel(query)
    print(f"Response: {result_without_lcel}")

    #  Option 2: Use LCEL (LangChain Expression Language) - Future Implementation
    print("\n--- Retrieval Chain with LCEL ---")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(f"Response: {result_with_lcel}")