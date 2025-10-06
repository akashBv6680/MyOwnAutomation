import os
import ssl
import time
import logging
import requests
from imapclient import IMAPClient
from email.message import EmailMessage
from smtplib import SMTP_SSL

# --- Configuration & Setup ---

# Set up basic logging for visibility in the GitHub Actions console
def setup_logging():
    """Configures basic logging settings."""
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

# Load configuration from environment variables (MANDATORY in GitHub Actions)
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") # This should be an App Password/Token
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama_service:11434/api/generate") # Use the Docker service name
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# --- Core LLM Functionality ---

def generate_response(original_email_content, sender_address):
    """
    Calls the Ollama API to generate a professional reply based on the email content.
    Includes the critical 'Content-Type: application/json' header fix.
    """
    logging.info(f"Generating LLM response for email from {sender_address}...")
    
    # Define a clear system prompt for the agent
    system_prompt = (
        "You are a professional and concise automated email agent. "
        "Your task is to draft a friendly, brief, and helpful response to the user's email. "
        "Do not include a salutation (like 'Hi [Name]') or a signature (like 'Best regards'). "
        "Only output the body of the reply email."
    )

    # Combine the system instruction and user prompt (the email content)
    prompt_message = f"Draft a professional reply to the following email:\n\n---\n{original_email_content}"

    # Construct the payload for the Ollama API call
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_message,
        "system": system_prompt,
        "stream": False,  # We want the full response at once
    }

    # CRITICAL FIX: Ensure proper headers are sent for JSON content
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Use a short timeout for the API call 
        response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=300) 
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        
        # Parse the JSON response
        data = response.json()
        
        # Extract the generated text
        generated_text = data.get("response", "").strip()
        
        if generated_text:
            logging.info("LLM response generated successfully.")
            return generated_text
        else:
            logging.warning("LLM generated an empty response.")
            return "Apologies, the automated agent could not generate a response at this time."

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Ollama API at {OLLAMA_URL}: {e}")
        return "Apologies, the automated agent encountered an internal error and could not generate a response."


# --- Email Sending Functionality ---

def send_reply(to_address, original_subject, body):
    """
    Sends the generated email reply using the configured SMTP server.
    """
    logging.info(f"Attempting to send reply to {to_address}...")
    
    # 1. Create the reply email message
    msg = EmailMessage()
    
    # Ensure the subject has the "Re:" prefix
    if not original_subject.lower().startswith("re:"):
        reply_subject = f"Re: {original_subject}"
    else:
        reply_subject = original_subject
        
    msg['Subject'] = reply_subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_address
    
    # Add a closing/signature for completeness
    full_body = f"{body}\n\n---\nAutomated Reply Agent"
    msg.set_content(full_body)

    # 2. Send the email via SMTP
    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        
        with SMTP_SSL(SMTP_SERVER, 465, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logging.info(f"Successfully sent reply email to {to_address}.")
        return True
    
    except Exception as e:
        logging.error(f"Failed to send email via SMTP: {e}")
        return False

# --- Main Logic ---

def main():
    """Main function to orchestrate the email agent."""
    setup_logging()

    if not all([EMAIL_ADDRESS, EMAIL_PASSWORD]):
        logging.error("Missing EMAIL_ADDRESS or EMAIL_PASSWORD environment variables. Exiting.")
        return

    logging.info(f"Connecting to IMAP server: {IMAP_SERVER}...")
    
    # 1. Connect to IMAP
    try:
        with IMAPClient(IMAP_SERVER) as client:
            client.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            logging.info("Successfully logged in to IMAP server.")
            
            # Select the INBOX
            client.select_folder('INBOX')
            
            # Search for unread emails (SENTSINCE: date is often unreliable, using UNSEEN is simpler)
            messages = client.search('UNSEEN')
            
            logging.info(f"Found {len(messages)} unread email(s). Processing...")

            if not messages:
                logging.info("No unread emails found. Job complete.")
                return

            # Fetch the required data for each message: structure, body, and headers
            response = client.fetch(messages, ['ENVELOPE', 'BODY.PEEK[TEXT]'])
            
            for msg_id, data in response.items():
                
                # Extract sender, subject, and body
                envelope = data.get(b'ENVELOPE')
                
                if not envelope or not envelope.subject:
                    logging.warning(f"Skipping message ID {msg_id}: Envelope or Subject missing.")
                    client.add_flags(msg_id, ['\Seen'])
                    continue

                # The Envelope structure gives us the sender's address easily
                sender = envelope.from_[0]
                sender_address = f"{sender.mailbox.decode()}@{sender.host.decode()}"
                subject = envelope.subject.decode()

                # Get the plain text body (using BODY.PEEK[TEXT] or trying other parts)
                raw_body = data.get(b'BODY[TEXT]')
                if raw_body:
                    email_content = raw_body.decode('utf-8', errors='replace').strip()
                else:
                    logging.warning(f"Skipping message ID {msg_id}: Could not extract text body.")
                    client.add_flags(msg_id, ['\Seen'])
                    continue
                
                logging.info(f"Processing message ID {msg_id}. Subject: '{subject}'")

                # 2. Generate the reply
                reply_body = generate_response(email_content, sender_address)

                # 3. Send the reply
                success = send_reply(sender_address, subject, reply_body)

                # 4. Mark the original email as read ONLY IF the reply was sent successfully
                if success:
                    client.add_flags(msg_id, ['\Seen'])
                    logging.info(f"Message ID {msg_id} marked as read.")
                else:
                    logging.error(f"Message ID {msg_id} was NOT marked as read due to sending failure.")

    except Exception as e:
        logging.critical(f"A critical error occurred in the IMAP connection or processing loop: {e}")
        
if __name__ == '__main__':
    # Add a small wait time to ensure Ollama is ready, even if healthcheck passed
    time.sleep(10) 
    main()
