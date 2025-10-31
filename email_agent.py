import os
import smtplib
import imaplib
import email
import ssl
import json
import re
import time
from email.message import EmailMessage
import requests

# --- Configuration ---
OLLAMA_URL = os.environ.get("OLLAMA_URL")
LLM_MODEL = os.environ.get("LLM_MODEL")
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# --- LLM Instructions ---
DATA_SCIENCE_KNOWLEDGE = """
You are a Senior Data Scientist specializing in ML, deep learning, and RAG chatbots. Analyze technical inquiries and provide actionable advice.
"""

AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are Akash BV, a Senior Data Scientist. Reply to technical emails with context-aware recommendations and sign off as 'Best regards,\\nAkash BV'."
)

RESPONSE_SCHEMA_JSON = {
    "is_technical": "True if the email mentions ML, deep learning, or RAG projects.",
    "simple_reply_draft": "For technical emails, provide specific advice and next steps.",
    "non_technical_reply_draft": "For non-technical emails, reply politely.",
    "request_meeting": "True if a meeting seems appropriate.",
    "meeting_suggestion_draft": "Suggest a video call to discuss the project."
}
RESPONSE_SCHEMA_PROMPT = json.dumps(RESPONSE_SCHEMA_JSON, indent=2)


def _send_smtp_email(to_email, subject, content):
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
        print(f"ERROR: Failed to send email: {e}")
        return False


def _fetch_latest_unread_email():
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return None, None, None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        _, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()
        if not ids:
            return None, None, None
        latest_id = ids[-1]
        mail.store(latest_id, '+FLAGS', '\\Seen')
        _, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)
        from_header = email_message.get("From", "")
        subject = email_message.get("Subject", "No Subject")
        from_match = re.search(r"<([^>]+)>", from_header)
        from_email = from_match.group(1) if from_match else from_header
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()
        return from_email, subject, body
    except Exception as e:
        print(f"ERROR: Failed to fetch email: {e}")
        return None, None, None


def _run_ai_agent(email_data):
    prompt = (
        f"{AGENTIC_SYSTEM_INSTRUCTIONS}\nKnowledge: {DATA_SCIENCE_KNOWLEDGE}\n"
        f"EMAIL SUBJECT: {email_data['subject']}\nEMAIL BODY: {email_data['body']}\n"
        f"Reply using this JSON schema:\n{RESPONSE_SCHEMA_PROMPT}"
    )
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        response_json = response.json()
        # The actual JSON content is a string within the 'response' key
        ai_response_str = response_json.get('response', '{}')
        return json.loads(ai_response_str)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"ERROR: AI request failed: {e}")
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
        print("ERROR: AI failed; no reply sent.")
        return

    is_technical = ai_output.get("is_technical", False)
    request_meeting = ai_output.get("request_meeting", False)
    
    final_subject = f"Re: {subject}"
    if is_technical:
        reply_draft = ai_output.get("meeting_suggestion_draft") if request_meeting else ai_output.get("simple_reply_draft")
    else:
        reply_draft = ai_output.get("non_technical_reply_draft")

    if not reply_draft:
        reply_draft = "Thank you for your message. We will get back to you shortly."

    print("ACTION: Sending reply...")
    if _send_smtp_email(from_email, final_subject, reply_draft):
        print("SUCCESS: Reply sent.")
    else:
        print("FAILURE: Reply failed.")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AI RUN COMPLETE ---")


if __name__ == "__main__":
    main_agent_workflow()
