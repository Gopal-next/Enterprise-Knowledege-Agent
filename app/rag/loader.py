from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()

    loader_class = LOADERS.get(ext)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        loader = loader_class(file_path)
        return loader.load()

    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{file_path}': {e}"
        ) from e