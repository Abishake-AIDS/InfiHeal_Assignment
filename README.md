# InfiHeal_Assignment

# Mental Health Chat Dataset Application

## Overview

This application leverages a dataset containing mental health-related questions and answers to build a similarity search engine. It allows users to input a question and retrieves similar questions from the dataset along with their associated answers and topics. The application uses natural language processing techniques to encode questions and find the most relevant ones based on a user's query.

## How It Works

1. *Dataset Loading*: The application loads a dataset from Hugging Face that includes mental health-related questions, answers, and topics.

2. *Text Encoding*: It uses the sentence-transformers library to encode question texts into embeddings. These embeddings capture the semantic meaning of the questions.

3. *Similarity Search*: The encoded question embeddings are stored in a FAISS index for efficient similarity search. Given a user query, the application finds the most similar questions based on cosine similarity.

4. *Retrieval*: For each similar question found, the application retrieves the associated answer and topic from the dataset and displays them.

## Features

- Load and preprocess a dataset containing mental health questions, answers, and topics.
- Encode question texts into high-dimensional vectors using sentence-transformers.
- Store and search embeddings using FAISS for fast similarity retrieval.
- Retrieve and display similar questions along with their answers and topics based on user queries.


