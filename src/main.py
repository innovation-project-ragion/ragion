from fastapi import FastAPI
from streamlit_chat import message
app = FastAPI()

# Simple health check
#@app.get("/")
#def hello_world():
 #   return {"message": "Hello, World!"}

message("My message")
message("Hello bot!", is_user=True)