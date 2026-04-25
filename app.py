import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
import torch # Ensure torch is imported for BERT model usage
import PyPDF2
from PIL import Image
import pytesseract
import io
import re
import urllib.request
import json
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
import pandas as pd

# --- Model Loading ---
@st.cache_resource
def load_rubric_model():
    rubric_model_path = "final_rubric_model/kaggle/working/final_rubric_model16"
    tokenizer = T5Tokenizer.from_pretrained(rubric_model_path)
    model = T5ForConditionalGeneration.from_pretrained(rubric_model_path)
    return tokenizer, model

@st.cache_resource
def load_criterion_model():
    criterion_model_path = "final_criterion_model/kaggle/working/final_criterion_model"
    tokenizer = BertTokenizer.from_pretrained(criterion_model_path)
    model = BertForSequenceClassification.from_pretrained(criterion_model_path)
    return tokenizer, model

rubric_tokenizer, rubric_model = load_rubric_model()
criterion_tokenizer, criterion_model = load_criterion_model()


from files.rubrics import generate_rubric_batch_

# Modified to generate rubrics in batch
def generate_rubric_batch(question_texts, total_marks_for_questions=None):
    # Use the globally loaded rubric_tokenizer and rubric_model
    if not isinstance(question_texts, list):
        question_texts = [question_texts] # Ensure it's a list for single question input compatibility

    input_texts = [f"generate rubric: {q_text}" for q_text in question_texts]

    # Tokenize the batch input
    inputs = rubric_tokenizer(input_texts, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Generate the output for the batch
    outputs = rubric_model.generate(
        inputs.input_ids,
        max_length=512,
        min_length=20,
        num_beams=4,
        early_stopping=True
    )

    generated_rubrics_raw = [rubric_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Enhance the generated rubrics using LLaMA API
    enhanced_rubrics_raw = []
    for idx, raw_rubric in enumerate(generated_rubrics_raw):
        total_m = total_marks_for_questions[idx] if total_marks_for_questions and len(total_marks_for_questions) > idx else None
        q_text = question_texts[idx]
        enhanced = generate_rubric_batch_(q_text, raw_rubric, total_m)
        enhanced_rubrics_raw.append(enhanced)
    
    all_structured_rubrics = []

    for idx, generated_rubric_raw_single in enumerate(enhanced_rubrics_raw):
        total_marks_for_question = total_marks_for_questions[idx] if total_marks_for_questions and len(total_marks_for_questions) > idx else None
        
        # Post-process the rubric for better readability and consolidation (same logic as before)
        points_raw = re.split(r'(\d+\.\s*)', generated_rubric_raw_single)

        parsed_points = []
        current_point_text = ""

        for item in points_raw:
            if re.match(r'\d+\.\s*', item):
                if current_point_text:
                    parsed_points.append(current_point_text.strip())
                current_point_text = item
            else:
                current_point_text += item
        if current_point_text:
            parsed_points.append(current_point_text.strip())

        consolidated_rubric = {}

        for point_text in parsed_points:
            marks_found = sum(int(m) for m in re.findall(r'\[(\d+)\s*mark(?:s)?\]', point_text))
            clean_text = re.sub(r'\[\d+\s*mark(?:s)?\]', '', point_text).strip()
            clean_text_without_leading_number = re.sub(r'^\d+\.\s*', '', clean_text)

            if clean_text_without_leading_number not in consolidated_rubric:
                consolidated_rubric[clean_text_without_leading_number] = {"marks": 0, "original_text": clean_text_without_leading_number}
            consolidated_rubric[clean_text_without_leading_number]["marks"] += marks_found

        points_with_marks = []
        for text, data in consolidated_rubric.items():
            points_with_marks.append({
                "original_text": data["original_text"],
                "model_marks": data["marks"]
            })

        structured_rubric = []
        if total_marks_for_question is not None and total_marks_for_question > 0 and len(points_with_marks) > 0:
            model_total_marks = sum(p["model_marks"] for p in points_with_marks)

            if model_total_marks == 0:
                marks_per_point = total_marks_for_question / len(points_with_marks)
                remainder = total_marks_for_question
                for i, p in enumerate(points_with_marks):
                    p["final_marks"] = int(marks_per_point)
                    remainder -= p["final_marks"]
                for i in range(int(remainder)):
                    points_with_marks[i]["final_marks"] += 1
            else:
                shares = []
                for p in points_with_marks:
                    share = (p["model_marks"] / model_total_marks) * total_marks_for_question
                    shares.append({"point": p, "share": share, "fractional_part": share - int(share)})
                assigned_marks_sum = 0
                for s in shares:
                    s["point"]["final_marks"] = int(s["share"])
                    assigned_marks_sum += s["point"]["final_marks"]
                remainder = total_marks_for_question - assigned_marks_sum
                shares.sort(key=lambda x: x["fractional_part"], reverse=True)
                for i in range(int(remainder)):
                    shares[i]["point"]["final_marks"] += 1

            for p in points_with_marks:
                structured_rubric.append({
                    "text": p["original_text"],
                    "marks": p.get("final_marks", 0)
                })
        else: # Fallback to original model marks if no total_marks_for_question or no points
            for text, data in consolidated_rubric.items():
                structured_rubric.append({
                    "text": data["original_text"],
                    "marks": data["marks"]
                })
        all_structured_rubrics.append(structured_rubric)
    
    return all_structured_rubrics

def evaluate_answer(student_answer, rubric_points):
    evaluation_details = []
    total_obtained_marks = 0

    # Ensure student_answer is not empty, otherwise return 0 marks for all points
    if not student_answer.strip():
        for i, point in enumerate(rubric_points):
            evaluation_details.append({
                "point_number": i + 1,
                "rubric_text": point["text"],
                "max_marks": point["marks"],
                "obtained_marks": 0,
                "feedback": "No answer provided by the student."
            })
        return 0, evaluation_details
    
    for i, point in enumerate(rubric_points):
        rubric_text = point["text"]
        point_marks = point["marks"]
        obtained_marks = 0
        feedback = "Not evaluated."

        if not rubric_text.strip():
            evaluation_details.append({
                "point_number": i + 1,
                "rubric_text": rubric_text,
                "max_marks": point_marks,
                "obtained_marks": 0,
                "feedback": "Rubric point is empty, cannot evaluate."
            })
            continue

        # Combine rubric point and student answer for BERT input
        # BERT model for sequence classification expects inputs like [CLS] text_a [SEP] text_b [SEP]
        inputs = criterion_tokenizer(rubric_text, student_answer, return_tensors="pt", truncation=True, padding=True)
        outputs = criterion_model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # Assuming BERT model outputs a score from 0-10 directly based on predicted_class_id
        # The id2label in criterion model config.json shows 0-10, so directly use this.
        score = predicted_class_id
        obtained_marks = min(score, point_marks) # Ensure obtained marks do not exceed max marks for the point

        # Provide feedback based on the obtained marks relative to max marks
        if obtained_marks >= point_marks:
            feedback = f"Excellent! Achieved {obtained_marks} out of {point_marks} for this criterion."
        elif obtained_marks >= point_marks * 0.75:
            feedback = f"Good. Achieved {obtained_marks} out of {point_marks}. Minor improvements possible."
        elif obtained_marks >= point_marks * 0.5:
            feedback = f"Fair. Achieved {obtained_marks} out of {point_marks}. Needs more detail or accuracy."
        else:
            feedback = f"Needs significant improvement. Achieved {obtained_marks} out of {point_marks}."

        total_obtained_marks += obtained_marks
        evaluation_details.append({
            "point_number": i + 1,
            "rubric_text": rubric_text,
            "max_marks": point_marks,
            "obtained_marks": obtained_marks,
            "feedback": feedback
        })

    return total_obtained_marks, evaluation_details

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_docx(file):
    document = Document(file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_questions(text):
    # This is a placeholder for question extraction logic.
    # A more sophisticated approach would involve NLP techniques
    # to identify question patterns (e.g., numbered lists, question marks, specific keywords).
    # For now, let's assume questions are numbered and separated by newlines.
    questions = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|\Z)', text, re.DOTALL)
    questions = [q.strip() for q in questions if q.strip()]
    if not questions:
        # Fallback if no numbered questions are found, treat each paragraph as a potential question
        questions = [q.strip() for q in text.split('\n\n') if q.strip()]
    return questions

def create_rubric_pdf(questions, generated_rubrics):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("<b>Generated Rubrics</b>", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    for i, question_text in enumerate(questions):
        question_key = f"question_{i}"
        rubric_structured = generated_rubrics.get(question_key, [])
        
        story.append(Paragraph(f"<b>Question {i+1}:</b> {question_text}", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))
        
        if rubric_structured:
            rubric_content = []
            for j, point in enumerate(rubric_structured):
                mark_text = f" [{point['marks']} mark(s)]" if point['marks'] > 0 else ""
                rubric_content.append(f"{j+1}. {point['text']}{mark_text}")
            story.append(Paragraph("<br/>".join(rubric_content), styles['Normal']))
        else:
            story.append(Paragraph("<i>Rubric not generated.</i>", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        
        if i < len(questions) - 1:
            story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer

def create_evaluation_pdf(questions, full_student_answer, evaluation_results, generated_rubrics):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom style for bold text within paragraphs
    styles.add(ParagraphStyle(name='BoldNormal', parent=styles['Normal'], fontName='Helvetica-Bold'))

    story = []
    story.append(Paragraph("<b>Student Answer Evaluation Results</b>", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Escape special characters in full_student_answer to prevent ReportLab parsing errors
    # Replace '&' first to avoid double-escaping subsequent replacements
    escaped_full_student_answer = full_student_answer.replace('&', '&')
    escaped_full_student_answer = escaped_full_student_answer.replace('<', '<')
    escaped_full_student_answer = escaped_full_student_answer.replace('>', '>')

    story.append(Paragraph("<b>Full Student Answer:</b>", styles['h3']))
    story.append(Paragraph(escaped_full_student_answer, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())

    for i, question_text in enumerate(questions):
        question_key = f"question_{i}"
        results = evaluation_results.get(question_key)
        rubric_for_display = generated_rubrics.get(question_key, [])

        story.append(Paragraph(f"<b>Question {i+1}:</b> {question_text}", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))

        if results:
            story.append(Paragraph(f"<b>Total Marks Obtained:</b> {results['total_obtained_marks']} / {results['max_possible_marks']}", styles['BoldNormal']))
            story.append(Spacer(1, 0.1 * inch))
            
            story.append(Paragraph("<b>Rubric & Feedback:</b>", styles['h3']))
            for j, point in enumerate(rubric_for_display):
                detail = next((d for d in results['evaluation_details'] if d['point_number'] == j+1), None)
                if detail:
                    story.append(Paragraph(f"- <b>Point {detail['point_number']}:</b> {detail['rubric_text']} "
                                        f"[{detail['obtained_marks']} / {detail['max_marks']} mark(s)]", styles['Normal']))
                    story.append(Paragraph(f"  Feedback: <i>{detail['feedback']}</i>", styles['Normal']))
                else:
                    mark_text = f" [{point['marks']} mark(s)]" if point['marks'] > 0 else ""
                    story.append(Paragraph(f"- {j+1}. {point['text']}{mark_text} (No specific evaluation data)", styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Paragraph("<i>Evaluation results not available for this question.</i>", styles['Normal']))
        
        if i < len(questions) - 1:
            story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer

def split_student_answers(full_answer_text, questions):
    student_answers_map = {}
    
    # Use re.findall to extract the numbered questions and their answers.
    # Pattern looks for a number followed by a dot, optional whitespace,
    # then captures everything non-greedily until the next numbered question or end of string.
    # re.DOTALL ensures '.' matches newlines.
    
    # This pattern captures the full answer block for each question.
    # Group 1: question number, Group 2: answer text
    # We need to make sure the answer text doesn't include the next question's number.
    answers_found = re.findall(r'(\d+)\.\s*(.*?)(?=\n\s*\d+\.|\Z)', full_answer_text, re.DOTALL)
    
    # Map the found answers to the question indices
    for q_num_str, answer_text in answers_found:
        try:
            q_num = int(q_num_str) - 1 # Convert to 0-indexed
            if 0 <= q_num < len(questions):
                student_answers_map[f"question_{q_num}"] = answer_text.strip()
        except ValueError:
            # Handle cases where the extracted "number" isn't a valid integer
            pass

    # Fallback if specific numbered answers aren't found for all questions:
    # If there's only one question, assign the entire text as its answer.
    if not student_answers_map and len(questions) == 1:
        student_answers_map["question_0"] = full_answer_text.strip()
    # If some questions are still unmapped, assign empty strings to them.
    # This ensures every question has an entry in the map, even if empty.
    for idx in range(len(questions)):
        q_key = f"question_{idx}"
        if q_key not in student_answers_map:
            student_answers_map[q_key] = "" # Assign empty string for unmapped questions
            
    return student_answers_map


def main():
    st.title("Rubric Generator")

    # Initialize session state for questions and marks if not already present
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'marks' not in st.session_state:
        st.session_state.marks = {}
    if 'generated_rubrics' not in st.session_state:
        st.session_state.generated_rubrics = {}
    if 'evaluation_results' not in st.session_state: # Store evaluation results
        st.session_state.evaluation_results = {}
    if 'full_student_answer_text' not in st.session_state: # Store full student answer for all questions
        st.session_state.full_student_answer_text = ""
    if 'student_answers_split' not in st.session_state: # Store split student answers
        st.session_state.student_answers_split = {}


    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Rubric Generation", "Evaluate Answers", "Analytical Dashboard"])

    with tab1:
        st.header("Rubric Generation")
        st.write("Upload a question paper (PDF or image) or enter a question below to generate grading rubrics.")

        # File uploader for question paper
        uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

        if uploaded_file is not None:
            file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
            st.write(file_details)

            extracted_text = ""
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type.startswith("image/"):
                extracted_text = extract_text_from_image(uploaded_file)

            if extracted_text:
                st.subheader("Extracted Text:")
                st.text_area("Raw text from file:", value=extracted_text, height=300)

                st.session_state.questions = extract_questions(extracted_text)
                # Initialize marks for new questions if not already set
                for i, question in enumerate(st.session_state.questions):
                    if f"question_{i}" not in st.session_state.marks:
                        st.session_state.marks[f"question_{i}"] = 10 # Default marks
                st.success(f"Extracted {len(st.session_state.questions)} questions. Adjust marks in the sidebar.")
            else:
                st.error("Could not extract text from the uploaded file.")

        # Sidebar for mark adjustment (remains in sidebar, but action button is in tab1)
        with st.sidebar:
            st.header("Adjust Marks per Question")
            if st.session_state.questions:
                for i, question in enumerate(st.session_state.questions):
                    st.session_state.marks[f"question_{i}"] = st.number_input(
                        f"Marks for Question {i+1}",
                        min_value=5,
                        max_value=20,
                        value=st.session_state.marks.get(f"question_{i}", 10), # Default to 10 if not set
                        key=f"marks_q_{i}"
                    )
                if st.button("Generate Rubrics for All Questions"):
                    st.session_state.generated_rubrics = {} # Clear previous rubrics
                    
                    question_texts_to_generate = []
                    total_marks_list = []
                    for i, question_text in enumerate(st.session_state.questions):
                        question_texts_to_generate.append(question_text)
                        total_marks_list.append(st.session_state.marks.get(f"question_{i}", 10))
                    
                    if question_texts_to_generate:
                        with st.spinner(f"Generating rubrics for {len(question_texts_to_generate)} questions..."):
                            # Use the new batch generation function
                            all_generated_rubrics = generate_rubric_batch(
                                question_texts_to_generate, total_marks_list
                            )
                            # Store the results back into session state
                            for i, rubric in enumerate(all_generated_rubrics):
                                st.session_state.generated_rubrics[f"question_{i}"] = rubric
                        st.success("Rubrics generated!")
                    else:
                        st.warning("No questions available to generate rubrics for.")
            else:
                st.info("Upload a file first to adjust marks for extracted questions.")

        st.markdown("---")
        # Display generated rubrics in the main area for Tab 1
        if st.session_state.generated_rubrics:
            st.subheader("Generated Rubrics for Extracted Questions:")
            # Use st.session_state.questions directly, as they are the source keys
            for i, question_text_display in enumerate(st.session_state.questions):
                st.write(f"**Question {i+1}:** {question_text_display}")
                rubric_to_format = st.session_state.generated_rubrics.get(f"question_{i}", [])
                formatted_output_list = []
                if isinstance(rubric_to_format, list):
                    for j, point in enumerate(rubric_to_format):
                        if point['marks'] > 0:
                            formatted_output_list.append(f"{j+1}. {point['text']} [{point['marks']} mark(s)]")
                        else:
                            formatted_output_list.append(f"{j+1}. {point['text']}")
                st.markdown("\n".join(formatted_output_list))
                st.markdown("---")
            
            if st.session_state.questions and st.session_state.generated_rubrics:
                if st.download_button(
                    label="Download All Rubrics as PDF",
                    data=create_rubric_pdf(st.session_state.questions, st.session_state.generated_rubrics),
                    file_name="generated_rubrics.pdf",
                    mime="application/pdf",
                    key="download_rubrics_pdf"
                ):
                    st.success("Rubrics downloaded as PDF!")
            
        elif st.session_state.questions and uploaded_file: # Only show if questions are loaded but rubrics not generated yet
            st.info("Click 'Generate Rubrics for All Questions' in the sidebar to see rubrics.")


        st.markdown("---")
        st.write("Alternatively, enter a single question below to generate a rubric:")
        question = st.text_area("Enter your question here:", height=150, key="single_question_input")

        # The single question generation should still use a non-batch approach or a batch of size 1
        if st.button("Generate Rubric for Single Question"):
            if question:
                with st.spinner("Generating rubric..."):
                    # For a single question, call generate_rubric_batch with a list of one question
                    # and retrieve the first (and only) rubric from the returned list.
                    generated_rubric_structured_list = generate_rubric_batch(question)
                    generated_rubric_structured = generated_rubric_structured_list[0] if generated_rubric_structured_list else []

                    formatted_single_rubric = []
                    for i, point in enumerate(generated_rubric_structured):
                        if point['marks'] > 0:
                            formatted_single_rubric.append(f"{i+1}. {point['text']} [{point['marks']} mark(s)]")
                        else:
                            formatted_single_rubric.append(f"{i+1}. {point['text']}")
 
                    st.subheader("Generated Rubric:")
                    st.markdown("\n".join(formatted_single_rubric))
            else:
                st.warning("Please enter a question to generate a rubric.")

    with tab2:
        st.header("Evaluate Student Answers")
        if st.session_state.questions and st.session_state.generated_rubrics:
            
            student_answer_upload_key = "student_answer_all_uploader_tab2"
            student_answer_text_area_key = "student_answer_all_manual_input_tab2"

            # Global file uploader for all questions
            full_student_answer_file = st.file_uploader(
                "Upload ONE file containing answers for ALL questions (PDF, Image, DOCX):",
                type=["pdf", "png", "jpg", "jpeg", "docx"],
                key=student_answer_upload_key
            )

            extracted_full_student_answer_text_from_upload = ""
            if full_student_answer_file is not None:
                if full_student_answer_file.type == "application/pdf":
                    extracted_full_student_answer_text_from_upload = extract_text_from_pdf(full_student_answer_file)
                elif full_student_answer_file.type.startswith("image/"):
                    extracted_full_student_answer_text_from_upload = extract_text_from_image(full_student_answer_file)
                elif full_student_answer_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    extracted_full_student_answer_text_from_upload = extract_text_from_docx(full_student_answer_file)

                if extracted_full_student_answer_text_from_upload:
                    st.session_state.full_student_answer_text = extracted_full_student_answer_text_from_upload
                    st.success("Text extracted from the full student answer file!")
                else:
                    st.error("Could not extract text from the uploaded full student answer file.")
            
            full_student_answer_manual_input = st.text_area(
                "Or enter the full student answer for ALL questions manually:",
                value=st.session_state.full_student_answer_text, # Use session state as initial value
                height=300,
                key=student_answer_text_area_key
            )

            # Update session state if manual input changes, or if file upload was cleared
            if full_student_answer_manual_input != st.session_state.full_student_answer_text and not full_student_answer_file:
                st.session_state.full_student_answer_text = full_student_answer_manual_input
                st.info("Manual answer input updated!")
            elif full_student_answer_file is None and st.session_state.full_student_answer_text and not extracted_full_student_answer_text_from_upload:
                # If file was removed and there's no extracted text, clear manual text too
                st.session_state.full_student_answer_text = ""
                st.session_state.student_answers_split = {}
                st.info("File removed, clearing manual input and split answers.")


            # Ensure student_answers_split is always updated based on full_student_answer_text if questions are present
            if st.session_state.full_student_answer_text and st.session_state.questions:
                current_split_answers = split_student_answers(
                    st.session_state.full_student_answer_text, st.session_state.questions
                )
                if current_split_answers != st.session_state.student_answers_split:
                    st.session_state.student_answers_split = current_split_answers
            elif not st.session_state.full_student_answer_text:
                if st.session_state.student_answers_split: # Only clear if it's not already empty
                    st.session_state.student_answers_split = {} # Clear if full answer is empty


            # Display split answers for verification
            if st.session_state.student_answers_split:
                st.subheader("Individual Student Answers (for verification):")
                for i, q_text in enumerate(st.session_state.questions):
                    q_key = f"question_{i}"
                    answer = st.session_state.student_answers_split.get(q_key, "No answer found for this question.")
                    with st.expander(f"Answer for Question {i+1}: {q_text}"):
                        st.text_area(f"Student Answer for Q{i+1}", value=answer, height=150, key=f"student_answer_display_{i}")


            if st.button("Evaluate All Questions"):
                # Check if there are questions, generated rubrics, and split student answers
                if st.session_state.questions and st.session_state.generated_rubrics and st.session_state.student_answers_split:
                    st.session_state.evaluation_results = {} # Clear previous results

                    for q_idx, question_text in enumerate(st.session_state.questions):
                        question_key = f"question_{q_idx}"
                        student_individual_answer = st.session_state.student_answers_split.get(question_key, "")
                        total_marks = st.session_state.marks.get(question_key, 10)
                        rubric_for_evaluation = st.session_state.generated_rubrics.get(question_key, [])

                        if not student_individual_answer.strip():
                            st.warning(f"No student answer found for Question {q_idx+1}. Skipping evaluation for this question.")
                            st.session_state.evaluation_results[question_key] = {
                                "total_obtained_marks": 0,
                                "evaluation_details": [],
                                "max_possible_marks": total_marks,
                                "message": "No student answer provided."
                            }
                            continue

                        if not rubric_for_evaluation:
                            st.warning(f"Rubric not generated for Question {q_idx+1}. Attempting to generate now...")
                            # Use generate_rubric_batch for on-the-fly generation for a single question
                            single_question_list = [question_text]
                            single_total_marks_list = [total_marks]
                            generated_rubric_for_single = generate_rubric_batch(
                                single_question_list, tokenizer, model, single_total_marks_list
                            )
                            rubric_for_evaluation = generated_rubric_for_single[0] if generated_rubric_for_single else []
                            st.session_state.generated_rubrics[question_key] = rubric_for_evaluation
                        
                        if rubric_for_evaluation:
                            with st.spinner(f"Evaluating answer for Question {q_idx+1} using semantic similarity..."):
                                total_obtained, details = evaluate_answer(student_individual_answer, rubric_for_evaluation)
                                st.session_state.evaluation_results[question_key] = {
                                    "total_obtained_marks": total_obtained,
                                    "evaluation_details": details,
                                    "max_possible_marks": sum(p["marks"] for p in rubric_for_evaluation)
                                }
                        else:
                            st.error(f"Cannot evaluate Question {q_idx+1}: No rubric available (even after attempted generation).")
                            st.session_state.evaluation_results[question_key] = {
                                "total_obtained_marks": 0,
                                "evaluation_details": [],
                                "max_possible_marks": total_marks,
                                "message": "No rubric available for evaluation."
                            }
                    st.success("Evaluation complete for all questions!")
                else:
                    st.warning("Please ensure questions are extracted, rubrics are generated, and student answers are provided and split.")

            # Display evaluation results if available
            if st.session_state.evaluation_results:
                st.markdown("### Detailed Evaluation Results:")
                for i, question_text in enumerate(st.session_state.questions):
                    question_key = f"question_{i}"
                    if question_key in st.session_state.evaluation_results:
                        results = st.session_state.evaluation_results[question_key]
                        st.write(f"**Question {i+1}:** {question_text}")
                        st.write(f"**Total Marks Obtained:** {results['total_obtained_marks']} / {results['max_possible_marks']}")

                        with st.expander(f"View Rubric and Feedback for Question {i+1}"):
                            rubric_for_display = st.session_state.generated_rubrics.get(question_key, [])
                            if isinstance(rubric_for_display, list):
                                for j, point in enumerate(rubric_for_display):
                                    detail = next((d for d in results['evaluation_details'] if d['point_number'] == j+1), None)
                                    if detail:
                                        st.markdown(f"- **Point {detail['point_number']}:** {detail['rubric_text']} "
                                                    f"[{detail['obtained_marks']} / {detail['max_marks']} mark(s)]")
                                        st.markdown(f"  - Feedback: *{detail['feedback']}*")
                                    else:
                                        st.markdown(f"- {j+1}. {point['text']} [{point['marks']} mark(s)] (No specific evaluation data)")
                            else:
                                st.markdown("Rubric not available or incorrectly formatted.")
                        st.markdown("---")
            
            if st.session_state.questions and st.session_state.evaluation_results:
                if st.download_button(
                    label="Download Evaluation Results as PDF",
                    data=create_evaluation_pdf(
                        st.session_state.questions,
                        st.session_state.full_student_answer_text, # Still use full text for PDF display
                        st.session_state.evaluation_results,
                        st.session_state.generated_rubrics
                    ),
                    file_name="evaluation_results.pdf",
                    mime="application/pdf",
                    key="download_evaluation_pdf"
                ):
                    st.success("Evaluation results downloaded as PDF!")

        else:
            st.info("Please go to 'Rubric Generation' tab, upload a question paper, and click 'Generate Rubrics for All Questions' in the sidebar to enable evaluation.")


    with tab3:
        st.header("Analytical Dashboard")

        if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
            # --- 1. Data Preparation ---
            all_details = []
            for q_key, results in st.session_state.evaluation_results.items():
                q_index = int(q_key.split('_')[-1])
                question_text = st.session_state.questions[q_index]
                for detail in results['evaluation_details']:
                    all_details.append({
                        "Question": f"Q{q_index + 1}",
                        "Question Text": question_text,
                        "Rubric Point": detail['rubric_text'],
                        "Obtained Marks": detail['obtained_marks'],
                        "Max Marks": detail['max_marks'],
                        "Feedback": detail['feedback']
                    })
            
            if all_details:
                df = pd.DataFrame(all_details)

                # --- 2. Overall Performance ---
                st.subheader("Overall Performance")
                total_obtained = df['Obtained Marks'].sum()
                total_max = df['Max Marks'].sum()
                overall_percentage = (total_obtained / total_max * 100) if total_max > 0 else 0
                
                # Custom styling for the metric
                st.markdown(f"""
                <div style="background-color:#262730; border-radius:10px; padding:20px;">
                <h3 style="color:white; text-align:center;">Overall Score</h3>
                <p style="color:#28a745; font-size:24px; text-align:center; font-weight:bold;">{total_obtained} / {total_max} ({overall_percentage:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)


                # --- 3. Performance by Question ---
                st.subheader("Performance by Question")
                per_question_performance = df.groupby('Question').agg(
                    Obtained=('Obtained Marks', 'sum'),
                    Max=('Max Marks', 'sum')
                ).reset_index()
                per_question_performance['Percentage'] = (per_question_performance['Obtained'] / per_question_performance['Max']) * 100
                
                st.bar_chart(per_question_performance.set_index('Question')['Percentage'])

                # --- 4. Areas for Improvement ---
                st.subheader("Areas for Improvement")
                
                # Find points where student scored less than 50% of the marks for that point
                df['Point Percentage'] = (df['Obtained Marks'] / df['Max Marks']).fillna(0) if 'Max Marks' in df.columns and df['Max Marks'].notna().all() and (df['Max Marks'] != 0).all() else 0.0
                improvement_areas = df[df['Point Percentage'] < 0.5]

                if not improvement_areas.empty:
                    st.warning("The student should focus on the following topics where performance was weak:")
                    # Group by question to show areas of improvement per question
                    for name, group in improvement_areas.groupby('Question Text'):
                        with st.expander(f"**For Question: {name}**"):
                            for _, row in group.iterrows():
                                st.markdown(f"- **Topic:** {row['Rubric Point']}")
                                st.markdown(f"  - *Scored {row['Obtained Marks']} out of {row['Max Marks']} marks.*")
                                st.markdown(f"  - **Feedback:** _{row['Feedback']}_")
                else:
                    st.success("Excellent performance! No specific areas for improvement identified.")
            else:
                st.info("Evaluation results are available but empty. Please check the evaluation process.")

        else:
            st.info("Please evaluate a student's answer in the 'Evaluate Answers' tab to see the analysis.")


if __name__ == "__main__":
    main()