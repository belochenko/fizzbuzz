from fastapi import FastAPI

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/summarize/")
async def say_hello(large_text: str):
    result = summarizer(large_text, max_length=250, min_length=30, do_sample=False)
    return {"result": result[0]['summary_text']}
