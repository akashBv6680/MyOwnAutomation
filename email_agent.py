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
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:3.8b-mini-4k-instruct-q4_1")
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") 
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# Stronger knowledge base prompt for project context
DATA_SCIENCE_KNOWLEDGE = """
You are a Senior Data Scientist. You help solve project statements about classification, transfer learning, model comparison, deployment, and RAG chatbot integration. 
For technical emails or client projects, always reply with specific suggestions, typical approaches, stepwise recommendations, and next actions (dataset needs, model building, deployment, chat integration). Mention ML tools if relevant (CNNs, Streamlit, RAG, transfer learning). 
Provide actionable and context-rich advice.
"""

AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are Akash BV, Senior Data Scientist. Always provide context-aware, actionable advice in technical/project replies. Review the client's problem carefully, suggest relevant ML/deployment approaches, and offer a discovery call if the project is serious. Always sign: 'Best regards,\\nAkash BV'."
)

RESPONSE_SCHEMA_JSON = {
    "is_technical": "True if the email contains any project statement, ML/DS problem, or technical inquiry; False otherwise.",
    "simple_reply_draft": "If technical/project: Summarize requirements and reply with context-specific advice (e.g., list next steps, suggest suitable ML models, tools, or chatbots for their scenario, propose a discovery call if desired).",
    "non_technical_reply_draft": "If non-project/general: Thank politely, acknowledge receipt, and offer to help if relevant details are shared.",
    "request_meeting": "True if there is a strong project intent or the sender appears ready to proceed.",
    "meeting_suggestion_draft": "Invite to a video discovery call, offering Monday/Wednesday/Friday 2-5PM IST as slots."
}
RESPONSE_SCHEMA_PROMPT = json.dumps(RESPONSE_SCHEMA_JSON, indent=2)

def _send_smtp_email(to_email, subject, content):
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
            print("DEBUG: Attempting to log into SMTP server...")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("DEBUG: Sent email successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to send email: {e}")
        return False

def _fetch_latest_unread_email():
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("CRITICAL ERROR: Missing credentials.")
        return None, None, None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()
        if not ids:
            print("STATUS: No unread emails.")
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
        print(f"DEBUG: Email from {from_email}, subject: {subject}.")
        return from_email, subject, body
    except Exception as e:
        print(f"ERROR: Failed to fetch email: {e}")
        return None, None, None

def _run_ai_agent(email_data):
    prompt = (
        f"{AGENTIC_SYSTEM_INSTRUCTIONS}\nKnowledge: {DATA_SCIENCE_KNOWLEDGE}\n"
        f"INCOMING EMAIL:\nFROM: {email_data['from_email']}\nSUBJECT: {email_data['subject']}\nBODY: {email_data['body']}\n"
        "Reply using the schema below as valid JSON:\n" + RESPONSE_SCHEMA_PROMPT
    )
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 128
        }
    }
    headers = {'Content-Type': 'application/json'}
    session = requests.Session()
    session.headers.update(headers)
    try:
        print("DEBUG: Starting Ollama call with keep-alive...")
        result = ""
        last = time.time()
        with session.post(OLLAMA_URL, json=payload, stream=True, timeout=1800) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    result += line.decode('utf-8') + "\n"
                if time.time() - last > 60:
                    print(f"[{time.strftime('%H:%M:%S')}] Still working...")
                    last = time.time()
        json_objects = re.findall(r'\{.*?\}', result, re.DOTALL)
        if json_objects:
            return json.loads(json_objects[-1])
        else:
            print("ERROR: No JSON found in Ollama response.")
            return None
    except Exception as e:
        print(f"ERROR: Ollama request failed: {e}")
        return None

def main_agent_workflow():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI RUN ---")
    from_email, subject, body = _fetch_latest_unread_email()
    if not from_email:
        print("STATUS: No new unread emails to process.")
        return
    email_data = {"from_email": from_email, "subject": subject, "body": body}
    ai_output = _run_ai_agent(email_data)
    if not ai_output:
        print("ERROR: AI failed (timeout/resource); no reply sent.")
        return
    is_technical = ai_output.get("is_technical", False)
    if isinstance(is_technical, str):
        is_technical = is_technical.lower() == "true"
    request_meeting = ai_output.get("request_meeting", False)
    if isinstance(request_meeting, str):
        request_meeting = request_meeting.lower() == "true"
    simple_reply_draft = ai_output.get("simple_reply_draft")
    non_technical_reply_draft = ai_output.get("non_technical_reply_draft")
    meeting_suggestion_draft = ai_output.get("meeting_suggestion_draft")
    print(f"RESULT: Technical? {is_technical} | Meeting? {request_meeting}")
    final_subject = f"Re: {subject}"
    reply_draft = (
        meeting_suggestion_draft if is_technical and request_meeting
        else simple_reply_draft if is_technical
        else non_technical_reply_draft
    )
    reply_draft = re.sub(r'<[^>]+>', '', str(reply_draft)).strip()
    if not reply_draft.lower().startswith(("hello", "hi", "thank you")):
        reply_draft = f"Hello,\n\n{reply_draft}"
    print("ACTION: Sending reply...")
    if _send_smtp_email(from_email, final_subject, reply_draft):
        print(f"SUCCESS: Reply sent.")
    else:
        print(f"FAILURE: Reply failed.")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AI RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
