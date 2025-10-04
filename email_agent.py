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
    os.environ["LANGCHAIN_PROJECT"] = "Agentic_Email_Langgraph_Sim_V2"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# --- System Prompts for Agentic Calls (STRICTEST FILTERING ENFORCED) ---

# Agents 1-4 (Drafting and Knowledge Integration) Instructions
AGENTIC_DRAFTING_INSTRUCTIONS = (
    "You are a professional, Agentic AI system acting ONLY as Senior Data Scientist, Akash BV. You are equipped with a vast Data Science Knowledge Base (RAG), simulating expert level understanding of ML/DL/Stats/Time Series.\n"
    "CRITICAL TONE: Your tone must be warm, highly conversational, and proactive. You MUST translate complex technical answers into **simple, easily understandable English** for non-technical clients. **AVOID JARGON, use analogies, and focus on business value.**\n\n"
    
    "**AGENT 1 (Condition Checker - PRIMARY GATEWAY):** Determine if the email is a specific project inquiry or technical question. The email is considered technical/project-related if it contains **ANY** of the following key terms or similar concepts, **no matter how short or brief the email is**:\n"
    " - **Project/Business Terms:** 'project details', 'ML Project', 'problem statement', 'Business use case', 'Ensure the problem', 'Insights', 'data strategy', 'modeling', 'data pipeline', 'predictive', 'ROI', 'optimization', 'forecasting'.\n"
    " - **Technical Terms:** 'Data Science', 'Machine Learning', 'Deep Learning', 'Statistical Modeling', 'Time Series', 'Data Engineering', 'Neural Network', 'Model performance', 'algorithm', 'data analysis', 'datasets', 'cloud computing'.\n"
    
    "If the email is purely administrative (e.g., 'Thank you,' 'Holiday schedule,' 'Meeting time confirmation'), generic, or non-technical (e.g., 'How are you?'), you **MUST** set 'is_ds_related' to False. **ABSOLUTELY DO NOT DRAFT A REPLY IF is_ds_related IS FALSE. THIS IS A HARD REQUIREMENT.**\n"
    
    "**AGENT 2 (Drafter/Translator):** For TECHNICAL queries, generate a full, highly conversational, and simple-English reply.\n"
    "**AGENT 3 (G-Meet Scheduler):** For TECHNICAL queries, proactively suggest a meeting time (Monday, Wednesday, or Friday between 2:00 PM and 5:00 PM IST) in the body of the reply.\n"
    "**AGENT 4 (Knowledge Base Integrator):** Use your simulated RAG knowledge to provide a concise, insightful technical comment or question related to the email topic, ensuring the explanation is in simple English.\n"
    
    "CRITICAL FORMATTING GUIDANCE:\n"
    " - All drafts MUST be in **PLAIN TEXT** format. **DO NOT USE HTML TAGS.**\n"
    " - All replies MUST be signed off with the exact signature: 'Best regards,\\nAkash BV'."
)

# Agent 5 (Approver/Refiner) Instructions
AGENTIC_APPROVER_INSTRUCTIONS = (
    "You are the Main Approver Agent (Agent 5), responsible for quality control. Your task is to review a draft reply against strict criteria. If the reply is not approved, provide precise, constructive feedback to the drafting agent.\n"
    "CRITERIA:\n"
    "1. **Tone/Clarity:** Is the language simple, warm, and highly conversational (suitable for a non-technical client)?\n"
    "2. **Technical Depth:** Does it contain a substantive technical comment/question (translated to simple English)?\n"
    "3. **Scheduling:** Does it clearly propose a meeting time (Mon, Wed, or Fri, 2-5 PM IST)?\n"
    "4. **Signature:** Does it use the correct sign-off: 'Best regards,\\nAkash BV'?\n\n"
    
    "If ALL criteria are met, set 'is_approved' to true. If any criterion is missed, set 'is_approved' to false and provide specific, actionable feedback in 'refinement_needed'. Use maximum 2 sentences for refinement_needed."
)

# --- Graph State Definition (Simulates Langgraph State) ---
EmailAnalysisState = {
    "from_email": str,
    "subject": str,
    "body": str,
    "is_ds_related": bool,
    "final_reply_draft": str,
    "refinement_needed": str, # New field for Agent 5 feedback
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


# --- Agent 1-4: Drafting Node ---

def drafting_agent_node(state):
    """
    NODE: Combined Agents 1-4. Checks condition, drafts reply, and integrates schedule/knowledge.
    """
    attempt = state.get('attempt', 1)
    print(f"NODE 1-4: Running Drafting Agent (Attempt {attempt})...")
    
    # Define the required JSON schema
    schema = {
        "type": "OBJECT",
        "properties": {
            "is_ds_related": {"type": "BOOLEAN", "description": "True if the email is a DS/ML/Tech query, False otherwise."},
            "technical_reply_draft": {"type": "STRING", "description": "A full, conversational, simple-English reply including the technical comment and meeting suggestion (ONLY generated if is_ds_related is TRUE)."},
        },
        "required": ["is_ds_related", "technical_reply_draft"]
    }
    
    # Contextualize the prompt with previous feedback (if it's a retry)
    feedback_context = ""
    if state["refinement_needed"]:
        feedback_context = f"\n\n--- PREVIOUS FEEDBACK FROM APPROVER (Agent 5) ---\nPLEASE REVISE THE DRAFT BASED ON THIS CRITIQUE: {state['refinement_needed']}\n-----------------------------------\n"

    full_email_content = (
        f"EMAIL CONTENT:\nFROM: {state['from_email']}\nSUBJECT: {state['subject']}\nBODY:\n{state['body']}\n\n"
        f"{feedback_context}"
    )

    ai_output = _run_ai_agent(
        system_prompt=AGENTIC_DRAFTING_INSTRUCTIONS, 
        user_query=full_email_content, 
        response_schema=schema,
        temperature=0.6 
    )

    if not ai_output:
        # Fallback Logic: Crucial for preventing KeyErrors
        print("DEBUG: LLM failed to return valid JSON or API call failed. Forcing is_ds_related=False.")
        state["is_ds_related"] = False # Treat as non-DS if we can't parse the LLM output
        state["final_reply_draft"] = ""
        return state
    
    # --- CRITICAL DEBUG LOGGING (To help diagnose filtering failures) ---
    is_ds_related = ai_output.get("is_ds_related", False)
    draft_snippet = ai_output.get("technical_reply_draft", "")
    print(f"DEBUG: Agent 1 Condition Check Result: is_ds_related={is_ds_related}")
    print(f"DEBUG: Draft Snippet: {draft_snippet[:100]}...")
    # --- END DEBUG LOGGING ---

    state["is_ds_related"] = is_ds_related
    state["final_reply_draft"] = draft_snippet
    
    # Reset refinement needed for the next Approver check
    state["refinement_needed"] = "" 
    state['attempt'] = attempt + 1
    
    return state

# --- Agent 5: Approver Node ---

def refine_and_approve_node(state):
    """
    NODE: Agent 5 (Approver/Refiner). Checks the draft and provides feedback or approval.
    """
    if not state["is_ds_related"]:
        # This check is redundant due to the main graph logic, but serves as a safety message
        print("NODE 5: Non-DS email detected. Skipping approval.")
        return state

    print("NODE 5: Running Approver Agent (Agent 5) to check draft quality...")
    
    schema = {
        "type": "OBJECT",
        "properties": {
            "is_approved": {"type": "BOOLEAN", "description": "True if the draft meets all criteria, False otherwise."},
            "refinement_needed": {"type": "STRING", "description": "Specific, actionable feedback for the drafting agent (Max 2 sentences), only if is_approved is False."},
        },
        "required": ["is_approved", "refinement_needed"]
    }
    
    review_query = (
        "Review the following draft reply based on the criteria provided in your system instructions. "
        "Draft to Review:\n"
        "------------------------------------\n"
        f"{state['final_reply_draft']}"
        "\n------------------------------------"
    )

    ai_output = _run_ai_agent(
        system_prompt=AGENTIC_APPROVER_INSTRUCTIONS, 
        user_query=review_query, 
        response_schema=schema,
        temperature=0.1 # Very low temp for deterministic check/approval
    )

    if not ai_output:
        # Emergency Approval if the Approver LLM fails to parse, to prevent infinite loops
        print("CRITICAL: Approver LLM failed. Emergency Auto-Approving Draft.")
        state["is_approved"] = True
        return state

    state["is_approved"] = ai_output.get("is_approved", False)
    state["refinement_needed"] = ai_output.get("refinement_needed", "")
    
    if not state["is_approved"]:
        print(f"APPROVAL STATUS: Refinement needed: {state['refinement_needed']}")
    else:
        print("APPROVAL STATUS: Draft Approved.")
        
    return state

# --- Main Workflow (Simulating Graph Execution) ---

def execute_agentic_graph():
    """Fetches email, executes the multi-agent graph with loop, and sends the reply."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI GRAPH RUN V2 ---")

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
        "refinement_needed": "",
        "is_approved": False,
        "attempt": 1 # To track retries
    }

    # 2. Main Agent Loop (Drafting and Approval)
    MAX_ATTEMPTS = 3
    
    for i in range(MAX_ATTEMPTS):
        # --- Step A: Drafting (Agents 1-4) ---
        state = drafting_agent_node(state)
        
        # --- Strict DS Filtering (User Requirement) ---
        if not state["is_ds_related"]:
            print("STATUS: Email is NON-DATA SCIENCE related or LLM failed to classify. **STRICTLY DISCARDING REPLY**.")
            # If Agent 1 decides the condition is NOT met, we STOP immediately.
            return

        # --- Step B: Approval (Agent 5) ---
        state = refine_and_approve_node(state)
        
        if state["is_approved"]:
            print(f"STATUS: Draft approved after {i + 1} attempt(s). Breaking loop.")
            break
            
        if i == MAX_ATTEMPTS - 1:
            print("STATUS: Max attempts reached. Sending final unapproved draft as a simple fallback.")
            # If max attempts reached, override with a safe, simple reply to prevent silence
            state["final_reply_draft"] = (
                "Hello,\n\nThank you for your detailed query. Due to unexpected system delays during drafting, "
                "I am sending this simple acknowledgment. I will follow up with the comprehensive technical "
                "details and scheduling options within the next 24 hours.\n\nBest regards,\nAkash BV"
            )
            state["is_approved"] = True # Force approval to exit loop

    # 3. Finalizer (Sending Email)
    if not state["is_approved"]:
         # Should not happen due to the MAX_ATTEMPTS fallback, but good to check
         print("FINAL FAILURE: Final state not approved. Aborting send.")
         return

    reply_draft = state["final_reply_draft"]
    final_subject = f"Re: {state['subject']}" 
    
    # Post-process cleanup
    reply_draft = re.sub(r'<[^>]+>', '', reply_draft).strip()
    if not reply_draft.lower().startswith(("hello", "hi", "dear", "thank you")):
         reply_draft = f"Hello,\n\n{reply_draft}"

    print(f"\nFINAL ACTION: Sending reply to {state['from_email']}...")
    
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
