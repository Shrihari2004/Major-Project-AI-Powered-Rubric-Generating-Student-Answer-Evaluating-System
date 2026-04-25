# AI-Powered Rubric Generating & Student Answer Evaluating System

This is a comprehensive web application built with Streamlit that leverages Machine Learning models to automate the grading process for educators. It can extract questions from various document formats, generate structured grading rubrics, and evaluate student answers against those rubrics.

## Features

- **Document Parsing:** Extracts questions from uploaded PDF, Image (PNG/JPG), and DOCX files using PyPDF2, pytesseract, and python-docx.
- **Automated Rubric Generation:**
  - Uses a fine-tuned **T5 model** to generate and structure grading rubrics into a concise 3-criterion format for extracted questions.
- **Student Answer Evaluation:**
  - Evaluates student answers against the generated rubric criteria using a fine-tuned **BERT sequence classification model**.
  - Provides detailed, point-by-point feedback and marks allocation.
- **Analytical Dashboard:** Visualizes overall student performance, performance per question, and highlights areas requiring improvement.
- **PDF Export:** Allows downloading of the generated rubrics and the final evaluation results as structured PDF documents.

## Technology Stack

- **Frontend:** Streamlit
- **Machine Learning:** PyTorch, Hugging Face Transformers (T5, BERT)
- **Document Processing:** PyPDF2, pytesseract, python-docx, ReportLab (for PDF generation)
- **Data Manipulation:** Pandas

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shrihari2004/Major-Project-AI-Powered-Rubric-Generating-Student-Answer-Evaluating-System.git
   cd Major-Project-AI-Powered-Rubric-Generating-Student-Answer-Evaluating-System
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   Make sure to install the required libraries. A typical command might look like:
   ```bash
   pip install streamlit transformers torch PyPDF2 Pillow pytesseract python-docx reportlab pandas
   ```
   *(Note: You must also have Tesseract-OCR installed on your system for image text extraction to work).*

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Folder Structure

- `app.py`: The main Streamlit application containing the UI, model loading, and evaluation logic.
- `files/rubrics.py`: Contains the logic for enhancing and formatting the generated rubrics.
- `final_rubric_model/`: Directory containing the fine-tuned T5 model weights (not tracked in Git due to size).
- `final_criterion_model/`: Directory containing the fine-tuned BERT model weights (not tracked in Git due to size).

## Usage Guide

1. **Rubric Generation Tab:** Upload a question paper (PDF/Image). Adjust the marks per question in the sidebar, and click "Generate Rubrics".
2. **Evaluate Answers Tab:** Upload a single document containing the student's answers for all questions. Click "Evaluate All Questions" to view the detailed evaluation and feedback.
3. **Analytical Dashboard Tab:** View the performance summary, score distribution, and personalized improvement areas based on the evaluation results.
