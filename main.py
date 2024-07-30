from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from rag import get_similar_answers_and_topics

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class SimilarQuestion(BaseModel):
    category: str
    related_question: str
    article: str

class SimilarQuestionsResponse(BaseModel):
    results: List[SimilarQuestion]


@app.post("/rag", response_model=SimilarQuestionsResponse)
def rag(query_text: QueryRequest):
    
    similar_questions, answers, topics = get_similar_answers_and_topics(query_text)
    results = [
        SimilarQuestion(
            category=topic,
            related_question=question,
            article=answer
        )
        for question, answer, topic in zip(similar_questions, answers, topics)
    ]
    
    return SimilarQuestionsResponse(results=results)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)