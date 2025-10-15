Victorian Author Predictor
A machine learning-powered web application that identifies Victorian-era authors from text snippets using deep learning neural networks. This project demonstrates end-to-end ML deployment — from trained models to a production-ready web interface.

Technical Overview
Built with FastAPI and TensorFlow, this application employs two Long Short-Term Memory (LSTM) neural network architectures to classify text from 50 different Victorian authors including Charles Dickens, the Brontë sisters, Oscar Wilde, and Thomas Hardy.

Key Features:

Dual Model Architecture: Implements both Simple LSTM and Bidirectional LSTM models for comparative analysis.

Real-time Predictions: FastAPI backend provides instant author predictions with confidence scores.

Pre-trained Models: Includes serialized Keras models (.h5), tokenizer, and label encoder for immediate deployment.

Interactive Web UI: Clean, responsive interface with test snippets for demonstration.

Tech Stack:

Backend: FastAPI, Uvicorn

ML Framework: TensorFlow 2.20, Keras 3.11

Data Processing: NumPy, scikit-learn

Frontend: Vanilla JavaScript with async/await for seamless predictions

Use Cases:

Educational: Study Victorian literature patterns and authorship attribution.

Research: Benchmark for stylometric analysis and NLP tasks.

Extension: Template for building custom text classification systems.

The application processes text through a trained tokenizer, pads sequences to 200 tokens, and returns predictions from both models, 
allowing users to compare different neural network approaches to the same classification problem. Perfect for demonstrating ML model deployment, 
API design, and full-stack integration skills.
