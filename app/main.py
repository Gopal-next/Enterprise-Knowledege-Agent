from fastapi import FastAPI

from api.upload import router as upload_router

app = FastAPI(title="Enterprise Knowledge Agent")

app.include_router(upload_router)

@app.get("/")
def home():
    return {
        "message": "Enterprise Knowledge Agent Running"
    }