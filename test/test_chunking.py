import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.loader import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = load_pdf("D:\Enterprise knowledge agent\Data\pdfs\employee_handbook.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(docs)


print("Total Chunks:", len(chunks))

# print("\nFirst Chunk:\n")

# print(chunks[0].page_content)


for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content)