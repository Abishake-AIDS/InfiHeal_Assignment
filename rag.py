
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load dataset
dataset = load_dataset("mpingale/mental-health-chat-dataset")
df = pd.DataFrame(dataset['train'])

# Ensure dataset contains the necessary columns
required_columns = ['questionText', 'answerText', 'topic']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Dataset must contain '{col}' column.")

# Extract unique questions, answers, and topics
unique_questions = df['questionText'].dropna().unique()
question_to_answer = dict(zip(df['questionText'], df['answerText']))
question_to_topic = dict(zip(df['questionText'], df['topic']))

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode unique question texts
question_embeddings = model.encode(unique_questions)

# Set up FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

# Function to find similar questions
def find_similar_questions(query_text, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return indices, [unique_questions[i] for i in indices[0]]

# Function to get answers and topics based on similar questions
def get_similar_answers_and_topics(query_text, top_k=5):
    indices, similar_questions = find_similar_questions(query_text, top_k)
    answers = [question_to_answer.get(question, "No answer found") for question in similar_questions]
    topics = [question_to_topic.get(question, "No topic found") for question in similar_questions]
    return similar_questions, answers, topics

