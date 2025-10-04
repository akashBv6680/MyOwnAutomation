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
# NOTE: Removed in this version for maximum execution speed, but kept variables for completeness.
langsmith_key = os.environ.get("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key 
    os.environ["LANGCHAIN_PROJECT"] = "Agentic_Email_Langgraph_Sim_V3_FAST"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# --- Master Agent System Prompt (Single Call = Fastest Reply) ---
AGENTIC_MASTER_INSTRUCTIONS = (
    "You are the **Master Agentic AI System**, acting as the Senior Data Scientist, Akash BV. Your goal is to generate a final, polished, approved email reply in a **single step**.\n"
    "You MUST follow ALL rules below and output a reply that meets ALL criteria immediately.\n\n"
    
    "**RULE 1: CLASSIFICATION (Technical Check)**\n"
    " - Determine if the email is a specific project inquiry or technical question. The email is technical if it contains **ANY** terms like: 'Data Science', 'Machine Learning', 'Time Series', 'algorithm', 'model performance', 'forecasting', 'project details', 'data strategy', or 'datasets'.\n"
    " - If the email is purely administrative (e.g., 'Thank you', 'Holiday schedule', 'Meeting time confirmation') or generic, you **MUST** set 'is_ds_related' to False.\n"
    
    "**RULE 2: DRAFTING & CRITERIA (If Technical)**\n"
    " - **Tone/Clarity:** Must be warm, highly conversational, and proactive. Use **simple English** and **AVOID JARGON**.\n"
    " - **Technical Depth:** Must include one concise, insightful technical comment or question related to the email topic, translated into simple English.\n"
    " - **Scheduling:** Must proactively suggest a meeting time (Monday, Wednesday, or Friday between 2:00 PM and 5:00 PM IST) in the body.\n"
    " - **Signature:** Must use the exact signature: 'Best regards,\\nAkash BV'.\n"
    " - **Format:** The reply MUST be in **PLAIN TEXT** (no HTML tags). The draft should be complete and ready to send.\n"
    
    "**ACTION:** Generate the JSON output. If 'is_ds_related' is False, the 'final_reply_draft' must be an empty string."
)

# --- Graph State Definition (Simulates Langgraph State) ---
EmailAnalysisState = {
    "from_email": str,
    "subject": str,
    "body": str,
    "is_ds_related": bool,
    "final_reply_draft": str,
}

# --- Shared LLM Utility Function (Node Execution Core) ---

def _run_ai_agent(system_prompt, user_query, response_schema, temperature=0.5):
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
        "temperature": temperature,
        "max_tokens": 2048,
        "response_mime_type": "application/json",
        "response_schema": response_schema
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOGETHER_API_KEY}'
    }
    
    # Exponential backoff retry loop
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
            raw_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
            print(f"CRITICAL AI AGENT ERROR: Failed to parse JSON: {e}. Raw Content: {raw_content[:150]}...")
            return None
    return None


# --- Master Agent Node (Single Step Execution) ---

def master_agent_node(state):
    """
    NODE: Executes classification, drafting, and final approval in ONE call for maximum speed.
    """
    print("NODE MASTER: Running Single-Call Master Agent for immediate decision and reply...")
    
    # Define the required JSON schema for the single output
    schema = {
        "type": "OBJECT",
        "properties": {
            "is_ds_related": {"type": "BOOLEAN", "description": "True if the email is a DS/ML/Tech query, False otherwise."},
            "final_reply_draft": {"type": "STRING", "description": "The full, conversational, simple-English, approved reply including the technical comment and meeting suggestion. Must be an empty string if is_ds_related is FALSE."},
        },
        "required": ["is_ds_related", "final_reply_draft"]
    }
    
    full_email_content = (
        f"EMAIL CONTENT:\nFROM: {state['from_email']}\nSUBJECT: {state['subject']}\nBODY:\n{state['body']}\n"
    )

    ai_output = _run_ai_agent(
        system_prompt=AGENTIC_MASTER_INSTRUCTIONS, 
        user_query=full_email_content, 
        response_schema=schema,
        temperature=0.5 # A balanced temperature for creative drafting and compliance
    )

    if not ai_output:
        print("DEBUG: LLM failed to return valid JSON or API call failed. Forcing is_ds_related=False.")
        state["is_ds_related"] = False
        state["final_reply_draft"] = ""
        return state
    
    # Extract results
    state["is_ds_related"] = ai_output.get("is_ds_related", False)
    state["final_reply_draft"] = ai_output.get("final_reply_draft", "")
    
    # Debugging
    print(f"DEBUG: Master Agent Condition Check Result: is_ds_related={state['is_ds_related']}")
    print(f"DEBUG: Draft Status: {'Generated' if state['final_reply_draft'] else 'Empty (Non-DS)'}")
    
    return state

# --- Main Workflow (Linear Execution) ---

def execute_agentic_graph():
    """Fetches email, executes the master agent once, and sends the reply immediately."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI GRAPH RUN V3 (FASTEST MODE) ---")

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

    # 2. Master Agent Call (The ONLY LLM interaction)
    state = master_agent_node(state)
        
    # 3. Finalizer (Sending Email)
    if not state["is_ds_related"]:
        print("STATUS: Email is NON-DATA SCIENCE related or Master Agent failed to classify. **STRICTLY DISCARDING REPLY**.")
        return

    reply_draft = state["final_reply_draft"]
    final_subject = f"Re: {state['subject']}" 
    
    if not reply_draft:
        print("FINAL FAILURE: Master Agent successfully classified the email as DS-related but failed to generate a draft. Aborting send.")
        return

    # *** NEW: Explicitly display the final approved draft before sending ***
    print("\n=======================================================")
    print(f"  AGENTIC AI: FINAL APPROVED REPLY DRAFT (PRE-SEND) to {state['from_email']}  ")
    print("=======================================================")
    print(reply_draft)
    print("=======================================================\n")
    
    # Post-process cleanup (Ensuring proper start and removing potential remnants)
    reply_draft = re.sub(r'<[^>]+>', '', reply_draft).strip()
    if not reply_draft.lower().startswith(("hello", "hi", "dear", "thank you")):
         reply_draft = f"Hello,\n\n{reply_draft}"

    print(f"FINAL ACTION: Sending reply to {state['from_email']}...")
    
    if _send_smtp_email(state["from_email"], final_subject, reply_draft):
        print(f"SUCCESS: Agentic reply sent.")
    else:
        print(f"FAILURE: Failed to send email to {state['from_email']}.")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI GRAPH RUN COMPLETE ---")

# --- Standard Email I/O Functions (Unchanged) ---

def _send_smtp_email(to_email, subject, content):
    """Utility to send an email via SMTP_SSL."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials missing. Cannot send email.")
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
        print("ERROR: Email credentials missing. Cannot fetch email.")
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
        from_email = from_match.group(1) if from_match else from_header.split()[-1].strip('<>')
        
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdispo:
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            body = email_message.get_payload(decode=True).decode(errors='ignore')
        
        return from_email, subject, body

    except Exception as e:
        print(f"CRITICAL IMAP ERROR: Failed to fetch email: {e}")
        return None, None, None

if __name__ == "__main__":
    execute_agentic_graph()
