# Transparent Credit Risk Engine

**Author:** Ronit Singha Roy

## Project Overview
Credit risk assessment is a critical function in banking and financial services. This project goes beyond standard classification by engineering an end-to-end machine learning workflow to predict credit default risk, paired with a Generative AI explainability module to provide transparent, human-readable explanations for every assessment.

The system evaluates applicant data and classifies them by risk, enabling organizations to:
* Reduce credit losses
* Improve approval decisions
* Understand the "why" behind automated decisions through AI-generated insights
* Automate credit evaluation at scale

## Business Problem
Financial institutions receive thousands of credit card applications daily. Manually assessing risk is time-consuming and prone to human error. Furthermore, traditional machine learning models often act as "black boxes," making it difficult to justify approval or rejection decisions to stakeholders and regulators.

This project solves these problems by:
* Automating risk assessment with high-speed predictive modeling APIs.
* Driving transparent decision science by translating complex mathematical outputs into plain English using Large Language Models.
* Supporting faster, highly compliant loan approvals.

## Project Workflow
Data Ingestion & Preprocessing → Automated Model Training → Predictive API Serving (FastAPI) → Generative AI Insight Generation (LangChain) → Front-End Dashboard Integration

## Key Features
* **Machine Learning Pipeline:** Engineered an end-to-end preprocessing and training workflow in Python to automate model training cycles for advanced credit default risk analytics.
* **Real-Time Inference API:** Developed a predictive modeling API utilizing FastAPI to serve real-time inferences and integrate seamlessly with front-end dashboards.
* **GenAI Explainability Module:** Architected a transparent decision-making layer leveraging LangChain and Llama 3.1 to generate dynamic, natural language explanations for every risk assessment.

## Technologies Used
* **Programming:** Python
* **API Framework:** FastAPI
* **Machine Learning & Data:** Scikit-Learn, Pandas
* **Generative AI & Orchestration:** LangChain, Llama 3.1, NVIDIA NIM

## Dataset Description
The dataset involves complex financial records utilized to train the predictive models. Features typically analyzed include:
* Credit history and scoring
* Income and employment data
* Payment history and debt ratios
* Loan amounts and applicant demographics

**Target Variable:** Credit Default Risk

## Model Evaluation & Business Use Cases
The system's performance ensures high accuracy in predicting credit risk while maintaining absolute transparency. 

**Ideal Users:**
* Banks and Credit Unions
* FinTech Startups
* Loan Providers and Underwriters
* Financial Risk Management Teams
