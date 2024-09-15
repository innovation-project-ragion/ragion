from fastapi import FastAPI
app = FastAPI()

# Simple health check
@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}
