import os
import smtplib
import imaplib
import email
import ssl
import json
import re
import time
from email.message import EmailMessage
from airllm import AirLLMInternLM  # Import the specific AirLLM model class

# --- Configuration & Secrets ---
LLM_MODEL_NAME = "internlm/internlm-20b"
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# --- LLM Instructions and Schema ---
DATA_SCIENCE_KNOWLEDGE = """
You are a Senior Data Scientist specializing in ML, deep learning, model comparison, deployment (Streamlit), and RAG chatbot integration. When replying to technical or project inquiries, analyze the requirements, suggest relevant ML/deployment approaches (CNN, transfer learning, metrics, app & chatbot development), and propose next steps like dataset review or scheduling meetings.
"""

AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are Akash BV, Senior Data Scientist. For any email about ML, deep learning, Streamlit, RAG, or a project statement, you must reply with context-aware, actionable recommendations. Always analyze and summarize the inquiry, propose first steps, offer technical advice, and suggest further information or meetings. Sign your reply as 'Best regards,\\nAkash BV'."
)

RESPONSE_SCHEMA_JSON = {
    "is_technical": "True if subject or body mentions ML, deep learning, project, model training, deployment, comparison, Streamlit, or RAG; False otherwise.",
    "simple_reply_draft": "If technical, reply with specific advice (summarize client's goals, suggest appropriate ML methods, mention CNN for image classification, transfer learning, Streamlit for deployment, RAG for chatbots). Offer to review datasets and schedule a meeting.",
    "non_technical_reply_draft": "If not technical, reply politely and invite the sender to share further details for technical/project advice.",
    "request_meeting": "True if the inquiry shows intent for project work or next steps.",
    "meeting_suggestion_draft": "Invite sender to a video call (Mon/Wed/Fri 2-5PM IST) for a deep-dive discussion."
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
    try:
        print("Initializing AirLLM with model:", LLM_MODEL_NAME)
        model = AirLLMInternLM(LLM_MODEL_NAME)
        
        prompt = (
            f"{AGENTIC_SYSTEM_INSTRUCTIONS}\nKnowledge: {DATA_SCIENCE_KNOWLEDGE}\n"
            f"INCOMING EMAIL:\nFROM: {email_data['from_email']}\nSUBJECT: {email_data['subject']}\nBODY: {email_data['body']}\n"
            "Reply using the schema below as valid JSON:\n" + RESPONSE_SCHEMA_PROMPT
        )
        
        input_text = [prompt]
        input_tokens = model.tokenizer(
            input_text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=2048, # Increased for larger context
            padding=True
        )

        print("Generating response with AirLLM...")
        generation_output = model.generate(
            input_tokens['input_ids'].cuda(),
            max_new_tokens=512,  # Allow for a detailed JSON response
            use_cache=True,
            return_dict_in_generate=True
        )

        response_text = model.tokenizer.decode(generation_output.sequences[0])
        
        # Extract JSON from the generated text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            print("ERROR: No JSON found in AirLLM response.")
            return None
            
    except Exception as e:
        print(f"ERROR: AirLLM request failed: {e}")
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
        print("SUCCESS: Reply sent.")
    else:
        print("FAILURE: Reply failed.")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AI RUN COMPLETE ---")


if __name__ == "__main__":
    main_agent_workflow()
