from fastapi import APIRouter, UploadFile, File
import os

router = APIRouter()

PDF_DIR = "data/pdfs"

os.makedirs(PDF_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(PDF_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename
    }


