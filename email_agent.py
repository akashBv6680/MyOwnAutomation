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

# ----------- Configuration ----------- #
# Use Ollama locally
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# Or use AirLLM local API (update URL & SDK as needed)
# AIRLLM_API = "http://localhost:5000/api/v1/generate"  # example; customize as per your setup
LLM_MODEL = os.environ.get("LLAMA_MODEL", "mistral:7b-instruct-q4_0")  # default to Ollama
USE_AIRLLM = False  # Set to true to switch to AirLLM

# Email credentials from secrets
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# SMTP & IMAP
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# ----------- Core Knowledge and Instructions -----------
DATA_SCIENCE_KNOWLEDGE = """
You are a Senior Data Scientist. You solve ML, DL, deployment, and RAG chatbot projects. Respond with focus on ML models, transfer learning, deployment tips, and chatbot guestions.
"""

SYSTEM_INSTRUCTIONS = (
    "Always provide context-aware, detailed, technical, and actionable responses to emails involving ML, deep learning, deployment, or RAG chatbots. Summarize the query and suggest next steps. Sign as 'Best regards,\\nAkash BV'."
)

RESPONSE_SCHEMA_JSON = {
    "is_technical": "True if email mentions ML, DL, deployment, or chat projects; False otherwise.",
    "simple_reply_draft": "For technical, provide specific suggestions, next steps, and advice.",
    "non_technical_reply_draft": "For general/non-technical, reply politely and invite more info.",
    "request_meeting": "True if temp reflects project discussion would benefit.",
    "meeting_suggestion_draft": "Invite to a team call (Mon/Wed/Fri 2-5PM IST)."
}
RESPONSE_SCHEMA_PROMPT = json.dumps(RESPONSE_SCHEMA_JSON, indent=2)

# ----------- Helper functions ----------- #
def send_email(to_email, subject, content):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(content)
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=ssl.create_default_context()) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print("Email send error:", e)
        return False

def fetch_latest_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        ids = messages[0].split()
        if not ids:
            return None, None, None
        latest_id = ids[-1]
        mail.store(latest_id, '+FLAGS', '\\Seen')
        _, data = mail.fetch(latest_id, "(RFC822)")
        email_msg = email.message_from_bytes(data[0][1])
        from_email = email_msg.get("From")
        subject = email_msg.get("Subject")
        if email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_msg.get_payload(decode=True).decode()
        return from_email, subject, body
    except Exception as e:
        print("Fetch email error:", e)
        return None, None, None

def call_airllm(prompt):
    # Replace with your actual AirLLM API call as needed
    if not USE_AIRLLM:
        return None
    try:
        response = requests.post("http://localhost:5000/api/v1/generate", json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "max_tokens": 512
        }, timeout=1800)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print("AirLLM API error:", e)
        return None

def run_llm(email_data):
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\nKnowledge: {DATA_SCIENCE_KNOWLEDGE}\n"
        f"EMAIL SUBJECT: {email_data['subject']}\nEMAIL BODY: {email_data['body']}\n"
        "Reply in JSON format as per schema."
    )
    if USE_AIRLLM:
        response_text = call_airllm(prompt)
    else:
        # Use Ollama API
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 256
            }
        }
        headers = {'Content-Type': 'application/json'}
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=1800) as response:
                response.raise_for_status()
                result = ""
                for line in response.iter_lines():
                    if line:
                        result += line.decode('utf-8')
                # Robust JSON extraction
                json_objs = re.findall(r'\{.*?\}', result, re.DOTALL)
                response_text = json_objs[-1] if json_objs else ""
        except Exception as e:
            print("Ollama API error:", e)
            response_text = ""
    # Parse response
    try:
        return json.loads(response_text)
    except:
        return {}

def main():
    print("Starting Email Automation...")
    from_email, subject, body = fetch_latest_email()
    if not from_email:
        print("No new emails.")
        return
    email_data = {"from_email": from_email, "subject": subject, "body": body}
    llm_response = run_llm(email_data)
    # Extract and decide reply
    is_technical = llm_response.get("is_technical", False)
    if isinstance(is_technical, str):
        is_technical = is_technical.lower() == "true"
    request_meeting = llm_response.get("request_meeting", False)
    if isinstance(request_meeting, str):
        request_meeting = request_meeting.lower() == "true"
    reply_text = ""
    if is_technical:
        reply_text = llm_response.get("simple_reply_draft", "Thank you for your email. Will get back to you.")
    else:
        reply_text = llm_response.get("non_technical_reply_draft", "Thank you for your email.")
    if not reply_text.lower().startswith(("hello", "hi", "thank you")):
        reply_text = f"Hello,\n\n{reply_text}"
    send_email(from_email, "Re: " + subject, reply_text)
    print("Reply sent.")

if __name__ == "__main__":
    main()
