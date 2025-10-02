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
# NOTE: The TOGETHER_API_KEY must be set in your GitHub Repository Secrets.
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") 
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") # Your 16-character App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"
# CHANGED MODEL for faster response (Mixtral-8x7B is highly optimized for chat)
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" 

# --- User-Defined Logic (Customize These!) ---
# EXPANDED CONDITION: More explicit technical topics for better filtering.
AUTOMATION_CONDITION = (
    "Does the incoming email contain a technical or complex question related to any of the following fields? "
    "1. Machine Learning (ML models, hyperparameter tuning, evaluation metrics, model deployment). "
    "2. Deep Learning (Neural Networks, CNNs, RNNs, NLP using transformers). "
    "3. Data Engineering (ETL pipelines, cloud data storage, workflow orchestration like Airflow). "
    "4. Statistical Analysis (Hypothesis testing, A/B testing, regression analysis, time series forecasting). "
    "5. Exploratory Data Analysis (EDA, visualization tools, data cleaning, feature engineering). "
    "If yes, the condition is met. Ignore general inquiries or non-technical requests."
)
# MODIFIED CONTEXT: Focuses on simple, clear, conversational English for clients and ensures the correct signature.
KNOWLEDGE_BASE_CONTEXT = (
    "You are an experienced Senior Data Scientist and Technical Support Agent. Your goal is to provide **simple, clear, and conversational advice in plain English, avoiding jargon and complex technical terms.** Always assume the recipient has no technical knowledge. Focus on clarity and actionable, easy-to-understand explanations. For inquiries outside the Data Science domain, politely state that you specialize in technical support for data analysis and ML only. You **MUST** sign off your reply using the exact signature: 'Best regards,\nAkash BV'."
)

# --- Helper Functions ---

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
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"ERROR: Failed to send email to {to_email}: {e}")
        return False

def _fetch_latest_unread_email():
    """Fetches the latest unread email details and marks it as read."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return None, None, None

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        
        # Search for unread emails
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            return None, None, None

        latest_id = ids[-1]
        # Store as read *before* processing to prevent re-processing in case of error
        mail.store(latest_id, '+FLAGS', '\\Seen') 
        
        status, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        # Extract sender's email
        from_header = email_message.get("From", "")
        subject = email_message.get("Subject", "No Subject")
        
        from_match = re.search(r"<([^>]+)>", from_header)
        from_email = from_match.group(1) if from_match else from_header
        
        # Extract email body
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
        
        return from_email, subject, body

    except Exception as e:
        print(f"CRITICAL IMAP ERROR: Failed to fetch email. Check your EMAIL_PASSWORD (App Password). Error: {e}")
        return None, None, None

def _run_ai_agent(email_data, condition, kb_context):
    """
    Calls the Together AI LLM for structured decision making with retry logic.
    Mixtral-8x7B is used here for faster, higher-quality structured output.
    """
    if not TOGETHER_API_KEY:
        print("ERROR: Together AI API Key is missing.")
        return None

    system_prompt = (
        "You are a professional Email Automation Agent. Analyze the email against the Condition. "
        "Your priority is to determine if the condition is met and draft a professional reply based on the knowledge base. "
        "You MUST output the structured format below."
        "\n\n--- STRUCTURED OUTPUT ---\n"
        "CONDITION_MET: [YES or NO]\n"
        "REPLY_DRAFT: [Your complete, professional drafted email reply content]\n"
        "--- END STRUCTURED OUTPUT ---"
    )

    user_query = (
        f"--- USER-DEFINED CONDITION ---\n{condition}\n\n"
        f"--- KNOWLEDGE BASE CONTEXT ---\n{kb_context}\n\n"
        f"--- INCOMING EMAIL CONTENT ---\n"
        f"FROM: {email_data['from_email']}\n"
        f"SUBJECT: {email_data['subject']}\n"
        f"BODY:\n{email_data['body']}"
    )
    
    messages_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages_payload,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOGETHER_API_KEY}'
    }
    
    # Retry with exponential backoff for robustness
    for i in range(3): 
        try:
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_json = response.json()
            return response_json['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {2 * (i + 1)} seconds...")
            elif response.status_code == 401:
                print("ERROR: Together API Key unauthorized. Check TOGETHER_API_KEY.")
                return None
            else:
                print(f"HTTP Error: {response.status_code}. Retrying in {2 * (i + 1)} seconds...")
            time.sleep(2 ** (i + 1)) # Exponential backoff: 2s, 4s, 8s
        except Exception as e:
            print(f"AI Agent failed with unexpected error: {e}")
            return None
    return None

def main_agent_workflow():
    """The main entry point for the scheduled job."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING EMAIL AGENT RUN ---")

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

    ai_output = _run_ai_agent(email_data, AUTOMATION_CONDITION, KNOWLEDGE_BASE_CONTEXT)

    if not ai_output:
        print(f"ERROR: AI Agent failed to produce output for {from_email}. Exiting.")
        return

    # --- Process AI Output ---
    # Relaxed matching to handle potential markdown formatting from the LLM
    condition_match = re.search(r"CONDITION_MET:\s*\[?(YES|NO)\]?", ai_output, re.IGNORECASE)
    # Uses non-greedy match ([\s\S]*?) to capture everything after REPLY_DRAFT:
    draft_match = re.search(r"REPLY_DRAFT:\s*([\s\S]*?)(?:--- END STRUCTURED OUTPUT ---|$)", ai_output, re.IGNORECASE)
    
    condition_met = condition_match.group(1).upper() if condition_match else "NO"
    reply_draft = draft_match.group(1).strip() if draft_match else "Could not generate a draft reply."
    
    print(f"AGENT RESULT: Condition Met? {condition_met}")

    # --- THIS IS THE CRITICAL LOGIC SECTION ---
    if condition_met == "YES":
        final_subject = f"Re: {subject}"
        # Prepend a professional greeting/closer if the draft looks incomplete
        if not reply_draft.startswith("Hi") and not reply_draft.startswith("Hello"):
             reply_draft = f"Hello,\n\n{reply_draft}"
        
        print("ACTION: Attempting to send automated reply...")
        # If the condition is YES, the email is sent using _send_smtp_email()
        if _send_smtp_email(from_email, final_subject, reply_draft):
            print(f"SUCCESS: Automated reply sent to {from_email}.")
        else:
            print(f"FAILURE: Failed to send email to {from_email}.")
    else:
        print("ACTION: Condition was NOT met. No email sent.")
    # --- END CRITICAL LOGIC SECTION ---
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENT RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
