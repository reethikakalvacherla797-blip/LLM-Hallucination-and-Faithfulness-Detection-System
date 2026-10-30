# LLM Hallucination and Faithfulness Detection System
An NLP-based system that detects whether AI-generated answers are supported, contradicted, or not established by the provided context using Natural Language Inference (NLI).

## 📌 Project Overview

Large Language Models (LLMs) can sometimes generate incorrect information or answers that are not supported by the provided context. This is commonly known as AI hallucination.

This project implements a detection system that evaluates AI-generated answers against a reference context using Natural Language Inference (NLI).

The system classifies answers into three categories:

- **Supported:** The answer is consistent with the reference context.
- **Contradicted:** The answer conflicts with the reference context.
- **Insufficient Evidence:** The reference context does not provide enough information to establish the answer.

## 🎯 Objectives

- Identify unsupported or contradictory AI-generated answers.
- Evaluate the faithfulness of answers against reference information.
- Provide an easy-to-use interface for testing answers.
- Help users understand whether an answer is supported by the supplied evidence.

## 🛠️ Technologies Used

- Python
- Natural Language Processing (NLP)
- Natural Language Inference (NLI)
- Hugging Face Transformers
- PyTorch
- Streamlit

## ⚙️ How It Works

1. The user enters a reference context.
2. The user provides an AI-generated answer.
3. The NLI model compares the answer with the context.
4. The system predicts whether the answer is supported, contradicted, or not established.
5. The result is displayed in the Streamlit interface.

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/reethikakalvacherla797-blip/LLM-Hallucination-Faithfulness-Detection.git
```


### 2. Navigate to the Project Folder

```bash
cd LLM-Hallucination-Faithfulness-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 🧪 Example

**Reference Context:**

The Sun is a star at the center of our solar system.

**AI-Generated Answer:**

The Sun is a planet.

**Expected Classification:** Contradicted

**Explanation:** The answer conflicts with the reference context, which identifies the Sun as a star.

## 📂 Project Structure

```text
LLM-Hallucination-Faithfulness-Detection/
│
├── app.py
├── detector.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Expected Outcome

The project provides an interactive application for evaluating the consistency of AI-generated answers with supplied context.

It helps demonstrate how NLP and Natural Language Inference can be used to assess answer faithfulness and identify potential hallucinations.

## ⚠️ Limitations

- Predictions depend on the NLI model's performance.
- Incorrect classifications are possible.
- The system evaluates answers against the supplied context; it does not independently verify every factual claim.
- A supported answer is not necessarily true in the real world.

## 🔮 Future Enhancements

- Add support for PDF and DOCX documents.
- Evaluate answers using multiple NLI models.
- Improve detection of subtle contradictions.
- Add confidence scores and detailed explanations.
- Integrate hallucination detection into a Retrieval-Augmented Generation (RAG) pipeline.
