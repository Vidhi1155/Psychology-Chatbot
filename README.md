
# **Psychology Chatbot** (using RAG Pipeline)






## Demo

[![Watch the demo](https://img.youtube.com/vi/SSyKqDVkPuc/0.jpg)](https://youtu.be/SSyKqDVkPuc)

An AI-powered psychology chatbot that provides mental health support and answers psychology-related questions using Retrieval-Augmented Generation (RAG) with Ollama and FAISS.




## **Table of Contents**
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Datasets Used](#datasets-used)
- [RAG Pipeline Overview](#rag-pipeline-overview)
- [Ollama Integration](#ollama-integration)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [File Structure](#file-structure)
- [Installation and How to Use](#installation-and-how-to-use)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)


## **Introduction**
This psychology chatbot is designed to provide accurate, empathetic responses to mental health and psychology-related queries. It combines the power of large language models with domain-specific knowledge through a Retrieval-Augmented Generation (RAG) pipeline, ensuring responses are both contextually relevant and grounded in authoritative psychology sources.

The system addresses the growing need for accessible mental health information while maintaining accuracy by:
- Retrieving relevant information from curated psychology datasets
- Generating natural, compassionate responses using Ollama's Mistral model
- Maintaining conversation context for personalized interactions
- Offering multiple input/output modalities (text, speech)
## **Key Features**
#### **1) Multi-modal Interaction**
- **Text Input/Output**: Traditional chat interface
- **Speech-to-Text**: Voice input capability
- **Text-to-Speech**: Audio responses for accessibility

#### **2) Advanced Retrieval System**
- Semantic search using FAISS vector database
- Hybrid scoring (cosine similarity + keyword overlap)
- Context-aware retrieval from diverse psychology sources

#### **3) Conversation Management**
- Persistent chat memory across sessions
- Context window of previous interactions
- Emotionally intelligent response generation
## **Technology Stack**
#### **1) Core Components**
- **Ollama**: For running the Mistral LLM locally
- **FAISS**: Facebook's vector similarity search
- **Sentence Transformers**: For embedding generation
- **PyTorch**: Deep learning framework

#### **2) Backend**
- **FastAPI**: High-performance API server
- **Flask**: Web interface server

#### **3) Supporting Libraries**
- **SpeechRecognition**: For voice input
- **pyttsx3**: For text-to-speech
- **NLTK**: Text processing utilities


## **Datasets Used**
The chatbot's knowledge comes from carefully curated psychology sources:

1. **NOBA Psychology Textbook**  
   Comprehensive academic content covering core psychology concepts

2. **Psychology Today Articles**  
   Current, practical mental health information

3. **NCERT Psychology Textbooks (Class 11-12)**  
   Foundational psychology concepts with Indian educational perspective

4. **Mental Health Conversation Dataset (Kaggle)**  
   Real-world mental health dialogues (https://www.kaggle.com/datasets)

5. **Mental Health FAQs (Kaggle)**  
   Common mental health questions and answers (https://www.kaggle.com/datasets)

All sources were processed and stored in a unified JSON format with title/content structure for consistent retrieval.

## **RAG Pipeline Overview**
The Retrieval-Augmented Generation pipeline works as follows:

1. **Query Processing**:
   - User input is received via text or speech
   - Converted to embedding using Sentence Transformer

2. **Vector Retrieval**:
   - FAISS index searches for most relevant documents
   - Top 3 chunks retrieved based on cosine similarity

3. **Context Augmentation**:
   - Retrieved documents combined with chat history
   - Formatted into prompt for LLM

4. **Response Generation**:
   - Ollama's Mistral model generates response
   - Response grounded in retrieved content

5. **Memory Storage**:
   - Q/A pair stored in conversation history
   - Persisted to JSON for future sessions

   
## **Ollama Integration**
The system uses Ollama to run the Mistral LLM locally, providing:

- **Privacy**: All processing happens locally
- **Customization**: Fine-tuned prompt engineering
- **Efficiency**: Optimized for psychology domain
## Model Architecture
![model](https://substack-post-media.s3.amazonaws.com/public/images/2206b95b-8aad-4b23-968e-a7d851a66c0f_1887x1582.png)
## **Evaluation Metrics**
The system was evaluated with:

1) **Semantic Similarity:**
* Average score: 0.83 (threshold 0.7)
* 83.33% of responses above threshold

2) **Retrieval Quality:**
* Top-3 accuracy: 92%
* Average cosine similarity: 0.74

3) **Language Quality:**
* Average perplexity: 17.39
* ROUGE-1: 0.41
* BLEU: 0.10
## **File Structure**
```bash
  my_project/
│
├── data/
│   ├── Converting data into unified data.ipynb
│   ├── website scraping.ipynb
│   ├── mental_health_conversation.csv
│   ├── Mental_Health_FAQs.csv
│   ├── NCERT_class11.txt
│   ├── NCERT_class12.txt
│   ├── NOBA_book1_content.txt
│   ├── psychology_today_articles.txt
│   ├── train.csv
│   └── unified_dataset.json
│
├── templates/
│   └── index.html
│
├── RAG_pipeline.ipynb
├── chatbot_eval_output.json
├── data_mapping.json
├── evaluation_set.json
├── faiss_index.bin
├── fastapi_server.py
├── flask_ui.py
├── requirements.txt
```

## **Installation and How to Use**

1) **Install python 3.10 version**

2) ####  **Install Ollama** (for windows)
* Go to: https://ollama.com/download
* Download and install the .msi installer.
* After installation, open Command Prompt to use it.

3) #### **Start the Ollama service**
* **On command prompt**
```bash
  ollama serve
```
This starts the background service required to run models.

* **Pull the Mistral model**
```bash
  ollama pull mistral
```
This will download the Mistral 7B model from Ollama's registry.

* **Run the Mistral model**
```bash
  ollama run mistral
```
To run Mistral and start chatting with it

4) #### **Create virtual environment to run jupyter notebook in python version 3.10**
Open another command prompt
```bash
  py -0
  Remove-Item -Recurse -Force myenv
  py -3.10 -m venv myenv
  myenv\Scripts\Activate
  python --version
  pip install jupyter
  python -m ipykernel install --user --name=myenv --display-name "Python 3.10 (myenv)"
  jupyter notebook
```
5) #### **Start Ollama server**
On another command prompt
```bash
  set OLLAMA_HOST=127.0.0.1:11435 && ollama serve
```
This command sets the Ollama server address to 127.0.0.1:11435 (your local machine) and then starts the Ollama server. It allows your psychology chatbot to connect locally to Ollama's AI models for generating responses.

6) #### **In jupyter notebook**
* Place all dataset files in correct directories
* **Open terminal in jupyter notebook**
```bash
  cd "C:\Users\Vidhi\ML Chatbot"      # use your own directory that you have created
  pip install ollama
  pip install sentence_transformers
  pip install uvicorn
  pip install faiss-cpu
  python fastapi_server.py
```
* **Open another terminal in jupyter notebook**
```bash
  cd "C:\Users\Vidhi\ML Chatbot"     # use your own directory that you have created
  pip install flask
  pip install SpeechRecognition
  pip install pyttsx3
  python flask_ui.py
```
Click "http://127.0.0.1:5000" appearing on your terminal

7) #### **Now Psychology chatbot is ready to use**






    
## **Future Improvements**
* Add user authentication
* Implement topic-based conversation routing
* Expand knowledge base with more sources
* Add multilingual support
* Implement sentiment analysis for empathetic responses
## **Acknowledgements**
* NOBA Project for open psychology textbooks
* Psychology Today for mental health articles
* NCERT for educational resources
* Kaggle community for datasets
* Ollama for the LLM infrastructure

