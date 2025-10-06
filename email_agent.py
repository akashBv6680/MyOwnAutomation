import os
import smtplib
import imaplib
import email
import ssl
import re
import time
import datetime
from email.message import EmailMessage

# --- Configuration & Secrets (Loaded from GitHub Environment Variables) ---
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") # Your 16-character App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# --- Keyword Filtering Configuration ---
# Emails containing any of these keywords will trigger the automated meeting reply.
PROJECT_KEYWORDS = [
    "data science", "machine learning", "ml project", "deep learning", 
    "ai project", "time series", "forecasting", "predictive model", 
    "computer vision", "nlp", "data engineering", "model deployment"
]

def _is_project_inquiry_by_keyword(subject, body):
    """Checks if the email contains any predefined project-related keywords."""
    content = f"{subject.lower()} {body.lower()}"
    
    for keyword in PROJECT_KEYWORDS:
        if keyword in content:
            print(f"DEBUG: Keyword found: '{keyword}'. Triggering meeting reply.")
            return True
    
    print("DEBUG: No project keywords found. Skipping automated reply.")
    return False

# --- Dynamic Date Calculation ---

def _get_next_meeting_dates():
    """
    Calculates the next three available meeting slots (Monday, Wednesday, Friday)
    starting from the day after the current execution.
    """
    # 0=Monday, 2=Wednesday, 4=Friday
    meeting_days = [0, 2, 4] 
    next_dates = []
    
    # Start checking from tomorrow
    current_date = datetime.date.today() + datetime.timedelta(days=1)
    
    while len(next_dates) < 3:
        if current_date.weekday() in meeting_days:
            # Append the date and time range for clarity
            next_dates.append(f"{current_date.strftime('%A, %B %d')} (2:00 PM - 5:00 PM IST)")
        current_date += datetime.timedelta(days=1)
        
    return next_dates

# Generate the dynamic meeting suggestions based on today's date
NEXT_MEETING_SLOTS = _get_next_meeting_dates()

# --- Hardcoded Reply Template (Your requested format) ---
MEETING_REPLY_TEMPLATE = f"""
Thank you for reaching out regarding the project details. We appreciate your interest and are happy to discuss further.

To define the scope and understand your data requirements, I recommend a quick 45-minute discovery call.

Could you please let us know your availability for a meeting on any of the following dates and times:
- {NEXT_MEETING_SLOTS[0]}
- {NEXT_MEETING_SLOTS[1]}
- {NEXT_MEETING_SLOTS[2]}

Looking forward to connecting with you.

Best Regards,
Akash BV
Data Scientist
"""

# --- Helper Functions (IMAP/SMTP) ---

def _send_smtp_email(to_email, subject, content):
    """Utility to send an email via SMTP_SSL."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials not available.")
        return False
        
    try:
        msg = EmailMessage()
        final_subject = subject if subject.lower().startswith('re:') else f"Re: {subject}"
        
        msg["Subject"] = final_subject
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
    """Fetches the latest unread email details and returns the IMAP object for final cleanup."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("CRITICAL ERROR: EMAIL_ADDRESS or EMAIL_PASSWORD not set in environment.")
        return None, None, None, None, None

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
            mail.logout()
            return None, None, None, None, None

        latest_id = ids[-1]
        
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
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', 'ignore')
                        break
                    except:
                        continue
        else:
            body = email_message.get_payload(decode=True).decode('utf-8', 'ignore')
            
        print(f"DEBUG: Successfully processed email from {from_email} (ID: {latest_id.decode()})")
        # Return the mail object and ID along with email data for final marking
        return from_email, subject, body, mail, latest_id 

    except Exception as e:
        print(f"CRITICAL IMAP ERROR: An unexpected error occurred during email fetching: {e}")
        # Ensure mail connection is closed if an error occurs before returning it
        if 'mail' in locals():
            mail.close()
            mail.logout()
        return None, None, None, None, None

def main_agent_workflow():
    """The main entry point for the scheduled job."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING CRON AGENT RUN ---")
    
    # Fetch email data and the IMAP connection variables
    from_email, subject, body, mail, latest_id = _fetch_latest_unread_email()

    if not from_email:
        print("STATUS: Exiting: No new unread emails found.")
        return

    # 1. Check condition using simple keyword filter
    is_project_inquiry = _is_project_inquiry_by_keyword(subject, body)
    
    reply_success = False

    if is_project_inquiry:
        # PATH 1: Project Inquiry -> Send Meeting Request (Hardcoded template)
        reply_draft = MEETING_REPLY_TEMPLATE.strip()
        print("ACTION: Project Inquiry detected by keywords. Sending meeting request.")
        
        # Sending the email
        reply_success = _send_smtp_email(from_email, subject, reply_draft)
        
    else:
        # PATH 2: Non-Project Inquiry -> No automated reply, but we still mark it as read 
        # to ensure the cron job only deals with truly new emails each run.
        print("ACTION: Non-Project Inquiry/General. No reply generated. Marking as read.")
        reply_success = True 

    # --- Final Cleanup ---
    if reply_success and mail and latest_id:
        # Mark as seen if we successfully sent a reply OR decided not to reply (non-project).
        try:
            # We must decode the ID if it's a bytes object, as returned by IMAP
            mail.store(latest_id, '+FLAGS', '\\Seen')
            print(f"STATUS: Email from {from_email} marked as \\Seen.")
        except Exception as e:
            print(f"ERROR: Could not mark email as seen: {e}")
        finally:
            mail.close()
            mail.logout()
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- CRON AGENT RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
