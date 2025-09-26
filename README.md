# arXiv Research Paper Summarization and RAG Q&A System

## Overview

This project demonstrates a complete pipeline for processing a dataset of arXiv research papers to build and evaluate text summarization models. It begins with data acquisition and cleaning, compares several pre-trained transformer models, fine-tunes a T5 model for improved performance, and concludes by implementing a simple Retrieval-Augmented Generation (RAG) system for question-answering on a custom document.

## Key Steps & Features

1.  **Data Acquisition:** Downloads the `arxivdataset` from KaggleHub containing metadata for thousands of research papers.
2.  **Data Preprocessing:**
    * Loads the initial JSON data into a Pandas DataFrame.
    * Cleans and extracts structured information such as author names, PDF links, and tags from raw string formats.
3.  **PDF Text Extraction:**
    * Takes a sample of 5000 papers from the dataset.
    * Validates PDF links and uses the `PyMuPDF` library to extract the full text from each paper.
    * Applies text cleaning functions to normalize whitespace and remove noise.
4.  **Exploratory Data Analysis (EDA):**
    * Calculates word counts for paper summaries and full texts.
    * Removes outliers based on text length to create a more uniform dataset for training.
5.  **LLM Summarization Comparison:**
    * Evaluates the zero-shot summarization performance of three pre-trained models: `t5-small`, `facebook/bart-large-cnn`, and `google/long-t5-tglobal-base`.
    * Uses **ROUGE scores** to benchmark their effectiveness. The `BART` model showed the best initial performance.
6.  **Fine-Tuning `t5-small`:**
    * Prepares the dataset for training by tokenizing inputs and targets.
    * Fine-tunes the `t5-small` model on the task of summarizing paper text.
    * The fine-tuned model demonstrates a significant improvement in ROUGE scores, outperforming all pre-trained models.
7.  **RAG Implementation:**
    * Builds a question-answering system using a user-uploaded PDF.
    * Chunks the document text and creates vector embeddings using `sentence-transformers`.
    * Builds a searchable vector index using **FAISS**.
    * When a user asks a question, the system retrieves the most relevant text chunk from the FAISS index and feeds it as context to the fine-tuned model to generate an answer.

## Results

* **Pre-trained Model Comparison:** Among the tested models, `facebook/bart-large-cnn` delivered the best summarization results out-of-the-box, achieving the highest ROUGE scores.
* **Fine-Tuning:** After fine-tuning the `t5-small` model, its performance improved dramatically, achieving a **ROUGE-1 score of 0.738** on a test sample, significantly higher than any of the pre-trained models.
* **RAG System:** The final RAG implementation successfully demonstrates how the fine-tuned model can be leveraged to create an intelligent document Q&A system.

## How to Use

1.  Ensure you have a Python environment (Google Colab with a GPU is recommended).
2.  Install the required libraries listed in the `pip install` cells within the notebook.
3.  Run the cells of the `GenAI.ipynb` notebook sequentially.
4.  For the RAG section at the end, you will be prompted to upload a PDF file for querying.

## Requirements

The main libraries used in this project are:
* `pandas`
* `kagglehub`
* `requests` & `pymupdf`
* `transformers` & `datasets`
* `accelerate` & `evaluate` (with `rouge_score`)
* `faiss-cpu` & `sentence-transformers`
* `spacy` & `pypdf`
* `matplotlib`
