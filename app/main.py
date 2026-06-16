from fastapi import FastAPI

from api.upload import router as upload_router

app = FastAPI(title="Enterprise Knowledge Agent")

app.include_router(upload_router)

@app.get("/")
def home():
    return {
        "message": "Enterprise Knowledge Agent Running"
    }


from fastapi import FastAPI

app = FastAPI(
    title="Enterprise Knowledge Agent"
)

@app.get("/")
def home():
    return {
        "message": "API Running"
    }


from app.api.routes.health import router as health_router

app.include_router(
    health_router
)