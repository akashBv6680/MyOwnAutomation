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

# --- Configuration & Secrets ---
# NOTE: All secrets MUST be set in your GitHub Repository Secrets.
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") 
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") # Your 16-character App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" 

# --- LangSmith Configuration (For External Tracing) ---
langsmith_key = os.environ.get("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key 
    os.environ["LANGCHAIN_PROJECT"] = "Agentic_Email_Langgraph_Sim"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# --- System Prompts for Agentic Calls ---

# All Agentic roles are unified into one call with structured output for efficiency.
AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are a professional, Agentic AI system acting ONLY as Senior Data Scientist, Akash BV. You MUST NOT impersonate anyone else. Your task is to analyze the email and provide a structured JSON output.\n"
    
    "**AGENT 1: CONDITION CHECKER** - Determine if the email is a specific project inquiry or technical question related to Data Science, Machine Learning, Deep Learning, Statistical Modeling, Time Series, or Data Engineering.\n"
    "**AGENT 2/3/4: TRANSLATOR/SCHEDULER** - Generate the final reply draft based on the routing decision.\n\n"
    
    "**IF TECHNICAL:** Provide a professional, value-add technical comment (based on your general LLM knowledge) and proactively suggest a meeting time (Monday, Wednesday, or Friday between 2:00 PM and 5:00 PM IST).\n"
    "**IF NON-TECHNICAL:** Provide a polite, general acknowledgement, stating you will review the email and get back to them shortly.\n\n"
    
    "CRITICAL FORMATTING GUIDANCE:\n"
    " - All drafts (technical_reply_draft and non_technical_reply_draft) MUST be in **PLAIN TEXT** format. **DO NOT USE HTML TAGS (like <br> or <b>)**.\n"
    " - All replies MUST be signed off with the exact signature: 'Best regards,\\nAkash BV'."
)

# --- Graph State Definition (Simulates Langgraph State) ---
# This dictionary structure is passed between nodes.
EmailAnalysisState = {
    "from_email": str,
    "subject": str,
    "body": str,
    "is_ds_related": bool,
    "final_reply_draft": str,
}

# --- Shared LLM Utility Function (Node Execution Core) ---

def _run_ai_agent(system_prompt, user_query, response_schema):
    """Handles the API call with retries and structured output."""
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
            print(f"CRITICAL AI AGENT ERROR: Failed to parse JSON: {e}")
            return None
    return None

# --- Graph Node Functions (Simulating Langgraph Nodes) ---

def check_condition_node(state):
    """
    NODE 1: Runs the Condition Checker Agent.
    Updates the state with the routing decision and two possible drafts.
    """
    print("NODE 1: Running Condition Checker and Draft Generation...")
    
    # Define the required JSON schema for the single LLM call
    schema = {
        "type": "OBJECT",
        "properties": {
            # Agent 1 (Condition Checker)
            "is_ds_related": {"type": "BOOLEAN", "description": "True if the email matches the Data Science/ML/Data Engineering condition, False otherwise."},
            
            # Agent 2/3/4 (Translator/Scheduler) - These are two mutually exclusive outputs
            "technical_reply_draft": {"type": "STRING", "description": "A professional reply including a technical comment and meeting suggestion (USED IF is_ds_related is TRUE)."},
            "non_technical_reply_draft": {"type": "STRING", "description": "A polite, professional acknowledgement for general emails (USED IF is_ds_related is FALSE)."},
        },
        "required": ["is_ds_related", "technical_reply_draft", "non_technical_reply_draft"]
    }
    
    # Combine all email info for the LLM prompt
    full_email_content = (
        f"FROM: {state['from_email']}\n"
        f"SUBJECT: {state['subject']}\n"
        f"BODY:\n{state['body']}\n\n"
    )

    ai_output = _run_ai_agent(
        system_prompt=AGENTIC_SYSTEM_INSTRUCTIONS, 
        user_query=full_email_content, 
        response_schema=schema
    )

    if not ai_output:
        # Fallback to a safe, non-technical default state
        print("CRITICAL: LLM failed. Using safe fallback draft.")
        return {
            "is_ds_related": False, 
            "final_reply_draft": "Hello,\n\nThank you for reaching out. I am currently reviewing your inquiry and will send a detailed, tailored response shortly. Best regards,\nAkash BV"
        }

    # Agent 1 decision
    is_ds_related = ai_output.get("is_ds_related", False)
    
    # Agent 2/3/4 content
    if is_ds_related:
        reply_draft = ai_output.get("technical_reply_draft", "Reviewing technical details. Will reply soon. Best regards,\nAkash BV")
        print("ROUTING EDGE: Decision is **TECHNICAL**.")
    else:
        reply_draft = ai_output.get("non_technical_reply_draft", "Reviewing general inquiry. Will reply soon. Best regards,\nAkash BV")
        print("ROUTING EDGE: Decision is **NON-TECHNICAL**.")

    # Update the state (simulating a state transition)
    state["is_ds_related"] = is_ds_related
    state["final_reply_draft"] = reply_draft
    
    return state

# --- Main Workflow (Simulating Graph Execution) ---

def execute_agentic_graph():
    """Fetches email, executes the single-node graph, and sends the reply."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI GRAPH RUN ---")

    # 1. Fetch Email (Initial State)
    from_email, subject, body = _fetch_latest_unread_email()

    if not from_email:
        print("STATUS: No new unread emails found to process. Exiting.")
        return

    # Initialize the Graph State
    state = {
        "from_email": from_email,
        "subject": subject,
        "body": body,
        "is_ds_related": False,
        "final_reply_draft": "",
    }

    # 2. Execute Node 1 (Condition Check & Draft Generation)
    final_state = check_condition_node(state)

    # 3. Finalizer (Sending Email)
    reply_draft = final_state["final_reply_draft"]
    final_subject = f"Re: {final_state['subject']}"
    
    # Post-process cleanup and greeting
    reply_draft = re.sub(r'<[^>]+>', '', reply_draft).strip()
    if not reply_draft.lower().startswith("hello") and not reply_draft.lower().startswith("hi") and not reply_draft.lower().startswith("thank you"):
         reply_draft = f"Hello,\n\n{reply_draft}"

    print(f"\nFINAL ACTION: Sending reply to {final_state['from_email']}...")
    
    if _send_smtp_email(final_state["from_email"], final_subject, reply_draft):
        print(f"SUCCESS: Automated reply sent. Condition Met: {final_state['is_ds_related']}.")
    else:
        print(f"FAILURE: Failed to send email to {final_state['from_email']}.")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI GRAPH RUN COMPLETE ---")

# --- Standard Email I/O Functions (Kept for completeness) ---

def _send_smtp_email(to_email, subject, content):
    """Utility to send an email via SMTP_SSL."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
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
        print(f"ERROR: Failed to send email via SMTP: {e}")
        return False

def _fetch_latest_unread_email():
    """Fetches the latest unread email details and marks it as read."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
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
        print(f"CRITICAL IMAP ERROR: Failed to fetch email: {e}")
        return None, None, None

if __name__ == "__main__":
    execute_agentic_graph()
