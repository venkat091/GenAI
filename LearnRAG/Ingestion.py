import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("Hello from learnrag!")
    loader = TextLoader("D:/Generative_AI/GenAI/LearnRAG/mdiumblog.txt", encoding ="utf8") # autodetect_encoding = True
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of text chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))

    



if __name__ == "__main__":
    main()
