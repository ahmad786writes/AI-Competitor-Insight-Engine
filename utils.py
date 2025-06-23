import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

def load_excel_text_clean(file):
    xl = pd.ExcelFile(file)
    full_text = ""
    
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, dtype=str)  # Read everything as text
        df.dropna(how='all', inplace=True)    # Drop rows where all values are NaN

        # Drop completely empty columns
        df.dropna(axis=1, how='all', inplace=True)

        # Remove useless header rows (e.g., those that are just labels or template text)
        df = df[~df.astype(str).apply(lambda x: x.str.contains("Signature|Instructions|discussed|Rating Info|Address|Company Website|Date", case=False, na=False)).any(axis=1)]

        # Add meaningful data only
        if not df.empty:
            full_text += f"\n### Sheet: {sheet_name}\n"
            full_text += df.to_string(index=False)
            full_text += "\n\n"

    return full_text

def vectorize_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def retrieve_docs(vectorstore, query):
    results = vectorstore.similarity_search(query)
    return "\n".join([doc.page_content for doc in results])
