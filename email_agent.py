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
    os.environ["LANGCHAIN_PROJECT"] = "Email_automation_langgraph_sim"
    print("STATUS: LangSmith tracing configured.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("STATUS: LANGCHAIN_API_KEY not found. LangSmith tracing is disabled.")

# --- System Prompts for Agentic Calls ---

# Agent 1 (Condition Checker / Tone Analyzer)
CONDITION_CHECKER_INSTRUCTIONS = (
    "You are a sophisticated routing agent. Your only task is to analyze the incoming email based on the strict condition provided and determine the next step.\n"
    "STRICT CONDITION: The email MUST be a specific project inquiry or a technical question related to Data Science, Machine Learning, Deep Learning, Statistical Modeling, Time Series, or Data Engineering.\n"
    "OUTPUT RULE: Respond STRICTLY in the JSON format provided."
)

# Agent 2 (Reply Generator / Translator / Scheduler)
REPLY_GENERATOR_INSTRUCTIONS = (
    "You are Senior Data Scientist, Akash BV. Your task is to draft the final email reply based on the provided context and routing decision.\n"
    "PERSONA RULE: Reply professionally, courteously, and in **plain text only (NO HTML tags like <br>)**.\n"
    "SIGNATURE RULE: Sign off every reply with the exact signature: 'Best regards,\\nAkash BV'.\n\n"
    
    "IF the email IS Data Science related and requires a meeting: Provide a brief, value-add technical comment and proactively suggest a meeting time (Monday, Wednesday, or Friday between 2:00 PM and 5:00 PM IST).\n"
    "IF the email IS NOT Data Science related: Provide a polite, general acknowledgement, stating you will get back to them shortly after review."
)

# --- Helper Functions (Standard Email I/O) ---

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
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
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
        
        return from_email, subject, body

    except Exception as e:
        print(f"CRITICAL IMAP ERROR: An unexpected error occurred: {e}")
        return None, None, None

def _run_ai_agent(system_prompt, user_query, response_schema):
    """
    Calls the Together AI LLM with structured output enforcement.
    Simulates a single Langgraph 'Node' execution.
    """
    if not TOGETHER_API_KEY:
        print("CRITICAL ERROR: Together AI API Key is missing. Cannot run agent.")
        return None

    messages_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

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
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_json = response.json()
            raw_json_string = response_json['choices'][0]['message']['content'].strip()
            return json.loads(raw_json_string)

        except requests.exceptions.RequestException as e:
            if response.status_code == 401:
                print("CRITICAL ERROR: Together API Key unauthorized. Check TOGETHER_API_KEY.")
                return None
            time.sleep(2 ** (i + 1)) 
        except Exception as e:
            print(f"CRITICAL AI AGENT ERROR (Node {system_prompt.splitlines()[0]}): Failed to parse JSON: {e}")
            return None
    return None

# --- Main Agentic Workflow (Simulating Langgraph Routing) ---

def main_agent_workflow():
    """
    The main workflow orchestrating the two agents and final email sending.
    This simulates the power of Langgraph by explicitly routing the state based on Agent 1's output.
    """
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI RUN ---")

    from_email, subject, body = _fetch_latest_unread_email()

    if not from_email:
        print("STATUS: No new unread emails found to process. Exiting.")
        return

    print(f"STATUS: Found new email from: {from_email} (Subject: {subject})")
    
    full_email_content = (
        f"FROM: {from_email}\nSUBJECT: {subject}\nBODY:\n{body}\n"
    )

    # --- NODE 1: Condition Checker (Agent 1) ---
    print("\n--- NODE 1: Running Condition Checker ---")
    
    checker_schema = {
        "type": "OBJECT",
        "properties": {
            "is_data_science_related": {"type": "BOOLEAN", "description": "True if the email is a specific technical Data Science/ML/Data Engineering inquiry, False otherwise."},
            "requires_meeting": {"type": "BOOLEAN", "description": "True if the tone suggests a serious project pitch or complex query warranting a discovery call, False otherwise."},
            "initial_analysis": {"type": "STRING", "description": "A brief internal note on why the decision was made."}
        },
        "required": ["is_data_science_related", "requires_meeting"]
    }

    routing_decision = _run_ai_agent(
        system_prompt=CONDITION_CHECKER_INSTRUCTIONS, 
        user_query=full_email_content, 
        response_schema=checker_schema
    )

    if not routing_decision:
        print("CRITICAL ERROR: Agent 1 (Condition Checker) failed. Exiting.")
        return

    is_ds_related = routing_decision.get("is_data_science_related", False)
    req_meeting = routing_decision.get("requires_meeting", False)

    print(f"ROUTER DECISION: DS Related? {is_ds_related} | Requires Meeting? {req_meeting}")

    # --- NODE 2: Reply Generator (Agent 2, 3, 4 logic combined) ---
    print("\n--- NODE 2: Running Reply Generator ---")

    reply_schema = {
        "type": "OBJECT",
        "properties": {
            "final_reply_draft": {"type": "STRING", "description": "The complete, professional, plain-text email reply to be sent."},
            "suggested_subject": {"type": "STRING", "description": "The new subject line for the reply (e.g., Re: Your Project Inquiry)"}
        },
        "required": ["final_reply_draft", "suggested_subject"]
    }
    
    # Pass the decision to the Generator Agent for contextual reply creation
    generator_query = (
        f"The incoming email is:\n{full_email_content}\n\n"
        f"--- AGENT 1 DECISION ---\n"
        f"IS DATA SCIENCE RELATED: {is_ds_related}\n"
        f"REQUIRES MEETING: {req_meeting}\n"
        f"--- TASK ---\n"
        f"Generate the FINAL, plain-text reply based on these decisions and the instructions."
    )

    reply_output = _run_ai_agent(
        system_prompt=REPLY_GENERATOR_INSTRUCTIONS, 
        user_query=generator_query, 
        response_schema=reply_schema
    )

    if not reply_output:
        print("CRITICAL ERROR: Agent 2 (Reply Generator) failed. Sending safe default.")
        reply_draft = "Hello,\n\nThank you for reaching out. I'm currently reviewing your inquiry and will send a proper, detailed response shortly. Best regards,\nAkash BV"
        final_subject = f"Re: {subject}"
    else:
        reply_draft = reply_output.get("final_reply_draft", "")
        final_subject = reply_output.get("suggested_subject", f"Re: {subject}")

    # --- FINALIZER (Email Sender) ---
    print("\n--- FINALIZER: Sending Email ---")
    
    # 1. Post-process cleanup (robustness against LLM format errors)
    reply_draft = re.sub(r'<[^>]+>', '', reply_draft).strip()

    # 2. Ensure greeting
    if not reply_draft.lower().startswith("hello") and not reply_draft.lower().startswith("hi") and not reply_draft.lower().startswith("thank you"):
         reply_draft = f"Hello,\n\n{reply_draft}"

    print(f"ACTION: Sending automated reply to {from_email}.")
    
    if _send_smtp_email(from_email, final_subject, reply_draft):
        print(f"SUCCESS: Automated reply sent with subject: {final_subject}.")
    else:
        print(f"FAILURE: Failed to send email to {from_email}.")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
