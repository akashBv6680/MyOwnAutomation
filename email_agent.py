import os
import smtplib
import imaplib
import email
import ssl
import json
import re
import requests
import time
from email.message import EmailMessage

# --- Configuration & Secrets (Loaded from GitHub Environment Variables) ---
# NOTE: All secrets (TOGETHER_API_KEY, EMAIL_ADDRESS, EMAIL_PASSWORD, LANGCHAIN_API_KEY) 
# MUST be set in your GitHub Repository Secrets.
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") 
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") # Your 16-character App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"
# Using Mixtral for speed and complex instruction following
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" 

# --- LangSmith Configuration for Tracing ---
langsmith_key = os.environ.get("LANGCHAIN_API_KEY")

if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key 
    os.environ["LANGCHAIN_PROJECT"] = "Email_automation_schedule"
    print("STATUS: LangSmith tracing configured.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("STATUS: LANGCHAIN_API_KEY not found. LangSmith tracing is disabled.")


# --- Knowledge Base & Persona Configuration ---

# **CRITICAL STEP: PASTE YOUR PDF CONTENT HERE.**
# --------------------------------------------------------------------------------
# Ensure the text below contains the Time Series keywords (LSTM, ARIMA, Forecasting)
# for the AI to recognize the email and provide a high-quality, relevant answer.
DATA_SCIENCE_KNOWLEDGE = """
# Data Science Project & Service Knowledge Base (Updated for Time Series)
#
# --------------------------------------------------------------------------------
## 1. Core Services Offered:
- **Predictive Modeling:** Advanced Regression, Time Series Forecasting (including **ARIMA**, SARIMA, Prophet, and **LSTM** for complex sequences).
- **Natural Language Processing (NLP):** Sentiment Analysis, Topic Modeling, Text Summarization, and custom Named Entity Recognition (NER).
- **Computer Vision:** Object Detection, Image Segmentation, and OCR solutions using CNNs (YOLO, ResNet).
- **MLOps and Deployment:** Model containerization (Docker), CI/CD pipelines, and hosting on AWS SageMaker, Azure ML, or GCP Vertex AI.
- **Data Engineering:** ETL pipeline development using Python/Pandas, Spark, and SQL optimization for large datasets.
- **Data Visualization & Reporting:** Interactive dashboards built with Streamlit, Tableau, and Power BI for executive summaries.

## 2. Guidance for Time Series Forecasting (Specific Reply Content):
- **ARIMA (and variations like SARIMA):** Excellent baseline for linear relationships and stationary data. Great for **short-term, stable predictions** and when interpretability is key. Requires data stationarity.
- **LSTM (Long Short-Term Memory Networks):** A type of recurrent neural network (RNN) superior for capturing complex, non-linear relationships, long-term dependencies, and memory in the data. Ideal for **long-term predictions** or highly volatile, non-stationary data (e.g., stock prices, sensor data). Requires more data and computation.
- **Recommendation:** If the data is complex, non-linear, or the prediction horizon is long, **prioritize LSTM**. For simple, short-term, or highly interpretable needs, **ARIMA is better**.

## 3. Standard Client Engagement Process:
1. **Initial Discovery Call (45 minutes):** Define the business problem, available data sources, and establish success metrics.
2. **Data Audit and Preparation (Phase 1):** Comprehensive review of data quality, feature engineering, and cleaning.
3. **Model Prototyping and Validation (Phase 2):** Iterative development, hyperparameter tuning, and cross-validation.
4. **Deployment and Handoff (Phase 3):** Integration of the final model into the client's infrastructure and comprehensive documentation/training.
5. **Post-Deployment Monitoring:** Quarterly performance reviews and model drift detection.

## 4. Availability for Meetings:
Available for 45-minute discovery calls on **Mondays, Wednesdays, and Fridays** between 2:00 PM and 5:00 PM **IST** (Indian Standard Time). Please propose two time slots within this window.
"""
# --------------------------------------------------------------------------------

# Agent 1 Condition: Determines if the email is technical enough for a specialized reply.
AUTOMATION_CONDITION = (
    "Does the incoming email contain a technical question or an explicit project inquiry/pitch related to Data Science, "
    "Machine Learning (ML), Deep Learning, Data Engineering, or advanced Statistical Analysis? "
)

# Agent 2 & 4 Persona: Defines reply style and meeting scheduling logic.
AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are a professional, Agentic AI system acting as Senior Data Scientist, Akash BV. Your task is to perform all required roles and provide a structured JSON output.\n"
    "1. CONDITION CHECK: Determine if the email is technical or a project pitch (based on the AUTOMATION_CONDITION).\n"
    "2. TRANSLATOR: Generate a technical reply if needed, or a polite general reply if not technical.\n"
    "3. TONE ANALYZER: If the email contains clear project details, a project pitch, or a serious inquiry, suggest a meeting by setting 'request_meeting' to true.\n\n"
    
    "GUIDANCE:\n"
    " - If 'is_technical' is TRUE, use 'simple_reply_draft' and potentially 'meeting_suggestion_draft'.\n"
    " - If 'is_technical' is FALSE, use 'non_technical_reply_draft'.\n"
    " - USE THE KNOWLEDGE BASE for drafting technical replies and meeting availability. \n"
    " - You MUST sign off all replies with the exact signature: 'Best regards,\\nAkash BV'."
)

# --- Helper Functions (No changes needed here) ---

def _send_smtp_email(to_email, subject, content):
    """Utility to send an email via SMTP_SSL."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials not available.")
        return False
    
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(content)
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            print(f"DEBUG: Attempting to log into SMTP server {SMTP_SERVER}...")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("DEBUG: Successfully logged in and sent message.")
        return True
    except smtplib.SMTPAuthenticationError:
        print("CRITICAL SMTP ERROR: Authentication failed. Is your EMAIL_PASSWORD a 16-character App Password?")
        return False
    except Exception as e:
        print(f"ERROR: Failed to send email to {to_email}: {e}")
        return False

def _fetch_latest_unread_email():
    """Fetches the latest unread email details and marks it as read."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("CRITICAL ERROR: EMAIL_ADDRESS or EMAIL_PASSWORD not set in environment.")
        return None, None, None

    try:
        print(f"DEBUG: Attempting to log into IMAP server {IMAP_SERVER}...")
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        print("DEBUG: IMAP login successful.")
        
        mail.select("inbox")
        
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            print("STATUS: IMAP search found no unread emails.")
            return None, None, None

        latest_id = ids[-1]
        mail.store(latest_id, '+FLAGS', '\\Seen') 
        
        status, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        from_header = email_message.get("From", "")
        subject = email_message.get("Subject", "No Subject")
        
        from_match = re.search(r"<([^>]+)>", from_header)
        from_email = from_match.group(1) if from_match else from_header
        
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdispo:
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()
        
        print(f"DEBUG: Successfully processed email from {from_email} with subject: {subject[:30]}...")
        return from_email, subject, body

    except imaplib.IMAP4.error as e:
        print(f"CRITICAL IMAP ERROR: Failed to fetch email. Check your App Password and IMAP settings. Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"CRITICAL IMAP ERROR: An unexpected error occurred during email fetching: {e}")
        return None, None, None

def _run_ai_agent(email_data):
    """
    Calls the Together AI LLM using a structured JSON schema to simulate the four agents.
    """
    if not TOGETHER_API_KEY:
        print("CRITICAL ERROR: Together AI API Key is missing. Cannot run agent.")
        return None

    if len(DATA_SCIENCE_KNOWLEDGE.strip()) < 50:
         print("WARNING: DATA_SCIENCE_KNOWLEDGE is likely empty or too short. AI response quality will suffer.")

    user_query = (
        f"--- TASK CONFIGURATION ---\n"
        f"CONDITION TO CHECK: {AUTOMATION_CONDITION}\n"
        f"KNOWLEDGE BASE (For context and reply): {DATA_SCIENCE_KNOWLEDGE}\n\n"
        f"--- INCOMING EMAIL CONTENT ---\n"
        f"FROM: {email_data['from_email']}\n"
        f"SUBJECT: {email_data['subject']}\n"
        f"BODY:\n{email_data['body']}\n\n"
        "Analyze the email and respond using the required JSON schema below. Ensure all replies are non-technical and professional. REMEMBER: Always generate a reply draft for both technical and non-technical cases."
    )
    
    messages_payload = [
        {"role": "system", "content": AGENTIC_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_query}
    ]

    # JSON Schema definition to enforce structured output
    response_schema = {
        "type": "OBJECT",
        "properties": {
            # Agent 1: Condition Checker 
            "is_technical": {"type": "BOOLEAN", "description": "True if the email matches the technical/project condition, False otherwise."},
            
            # Agent 2a: Translator/Analyzer (Technical Reply)
            "simple_reply_draft": {"type": "STRING", "description": "The primary reply to the client, simplified and non-technical, based on the knowledge base (USED IF is_technical is TRUE)."},
            
            # Agent 2b: Translator/Analyzer (General Reply) -- NEW FIELD FOR NON-TECHNICAL RESPONSE
            "non_technical_reply_draft": {"type": "STRING", "description": "A polite, professional acknowledgement and offer to help, used if is_technical is FALSE (e.g., 'Thanks for reaching out, how can I help?')."},

            # Agent 4: Meeting Scheduler
            "request_meeting": {"type": "BOOLEAN", "description": "True if the tone suggests a serious project inquiry or pitch, False otherwise. (Triggers meeting suggestion)."},
            "meeting_suggestion_draft": {"type": "STRING", "description": "If request_meeting is true, draft a reply suggesting available dates from the knowledge base (e.g., 'Are you available this week on Monday, Wednesday, or Friday afternoon?')."},
        },
        "required": ["is_technical", "simple_reply_draft", "non_technical_reply_draft", "request_meeting", "meeting_suggestion_draft"]
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages_payload,
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_mime_type": "application/json",
        "response_schema": response_schema
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOGETHER_API_KEY}'
    }
    
    # Retry with exponential backoff for robustness
    for i in range(3): 
        try:
            print(f"DEBUG: Attempting Together AI API call (Retry {i+1})...")
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_json = response.json()
            raw_json_string = response_json['choices'][0]['message']['content'].strip()
            print("DEBUG: AI call successful. Attempting JSON parse...")
            return json.loads(raw_json_string)

        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                print(f"ERROR: Rate limit exceeded. Retrying in {2 * (i + 1)} seconds...")
            elif response.status_code == 401:
                print("CRITICAL ERROR: Together API Key unauthorized. Check TOGETHER_API_KEY.")
                return None
            else:
                print(f"HTTP Error: {response.status_code}. Retrying in {2 * (i + 1)} seconds...")
            time.sleep(2 ** (i + 1)) 
        except Exception as e:
            print(f"CRITICAL AI AGENT ERROR: Failed with unexpected error or failed to parse JSON: {e}")
            return None
    return None

def main_agent_workflow():
    """The main entry point for the scheduled job."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI RUN ---")

    from_email, subject, body = _fetch_latest_unread_email()

    if not from_email:
        print("STATUS: No new unread emails found to process. Exiting.")
        return

    print(f"STATUS: Found new email from: {from_email} (Subject: {subject})")
    
    email_data = {
        "from_email": from_email,
        "subject": subject,
        "body": body
    }

    ai_output = _run_ai_agent(email_data)

    if not ai_output:
        print(f"CRITICAL ERROR: Agentic AI failed to produce structured output for {from_email}. Exiting.")
        return

    # Extract results from the JSON output
    is_technical = ai_output.get("is_technical", False)
    request_meeting = ai_output.get("request_meeting", False)
    
    # Get all three potential reply drafts
    simple_reply_draft = ai_output.get("simple_reply_draft", "Error: Technical draft missing.")
    non_technical_reply_draft = ai_output.get("non_technical_reply_draft", "Thank you for your email. I'll review this and get back to you shortly. Best regards,\nAkash BV")
    meeting_suggestion_draft = ai_output.get("meeting_suggestion_draft", simple_reply_draft)
    
    # This is the most important log line! Check what the agent decided.
    print(f"AGENT RESULT: Is Technical/Project? {is_technical} | Request Meeting? {request_meeting}")

    final_subject = f"Re: {subject}"
    reply_draft = ""
    action_log = ""

    if is_technical:
        # TECHNICAL PATH (Use simple_reply_draft or meeting_suggestion_draft)
        if request_meeting:
            reply_draft = meeting_suggestion_draft
            action_log = "Condition met AND tone required meeting. Sending meeting suggestion."
        else:
            reply_draft = simple_reply_draft
            action_log = "Condition met. Sending simple technical explanation."
    else:
        # NON-TECHNICAL PATH (ALWAYS REPLY using non_technical_reply_draft)
        reply_draft = non_technical_reply_draft
        action_log = "Condition NOT met (General Inquiry). Sending polite, non-technical acknowledgement."
    
    # Prepend greeting if missing and attempt to send the email
    if not reply_draft.lower().startswith("hello") and not reply_draft.lower().startswith("hi"):
         reply_draft = f"Hello,\n\n{reply_draft}"
        
    print(f"ACTION: {action_log}")
    print("ACTION: Attempting to send automated reply...")
    
    if _send_smtp_email(from_email, final_subject, reply_draft):
        print(f"SUCCESS: Automated reply sent to {from_email}.")
    else:
        print(f"FAILURE: Failed to send email to {from_email}.")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
