import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import smtplib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import time # Added for timing trace runs
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from imblearn.over_sampling import SMOTE
from matplotlib.backends.backend_pdf import PdfPages
import ssl
import imaplib
import email

# --- LangSmith Setup ---
from langsmith import traceable, Client

# IMPORTANT: Ensure LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, and LANGCHAIN_TRACING_V2=true 
# are set as environment variables in your Streamlit secrets or deployment environment.
try:
    # Initialize the LangSmith client. This is mainly for manual tracing, 
    # but the environment variables enable auto-tracing via @traceable.
    client = Client()
    print("LangSmith Client initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize LangSmith Client (check ENV vars): {e}")

# === Email Config ===
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# === Streamlit Config ===
st.set_page_config(page_title="⚡ Fast AutoML Agent", layout="wide")
st.title("🤖 Fast AutoML + Email Agent")

# === Traceable SMTP Helper ===

@traceable(run_type="tool")
def _send_smtp_email(msg: EmailMessage):
    """
    Handles the actual SMTP communication and logs the network call to LangSmith.
    """
    recipient = msg["To"]
    subject = msg["Subject"]
    start_time = time.time()
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        duration = time.time() - start_time
        # Return structured output for tracing
        return {
            "status": "sent",
            "recipient": recipient,
            "subject": subject,
            "duration": round(duration, 3)
        }
    except Exception as e:
        # The exception is automatically recorded by @traceable
        raise RuntimeError(f"SMTP failed for {recipient} with subject '{subject}'") from e


# === Upload and Detect Task ===
st.sidebar.header("📊 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx"])

def is_valid_email(email_str):
    return re.match(r"[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$", email_str)

def generate_visualizations(df):
    pdf_name = "visual_report.pdf"
    with PdfPages(pdf_name) as pdf:
        for col in df.columns:
            plt.figure(figsize=(6, 4))
            if df[col].dtype == "object" or df[col].nunique() < 10:
                if df[col].nunique() <= 5:
                    df[col].value_counts().plot.pie(autopct='%1.1f%%')
                    plt.title(f"Pie Chart - {col}")
                else:
                    sns.countplot(y=col, data=df)
                    plt.title(f"Bar Chart - {col}")
            elif np.issubdtype(df[col].dtype, np.number):
                if df[col].nunique() < 20:
                    sns.histplot(df[col], kde=False)
                    plt.title(f"Hist Plot - {col}")
                else:
                    sns.kdeplot(df[col], fill=True)
                    plt.title(f"Dist Plot - {col}")
            else:
                continue
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    return pdf_name

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)
    st.write("✅ Data Preview")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("Select Target Column", df.columns)

    task_type = "classification" if df[target].nunique() <= 20 or df[target].dtype == 'object' else "regression"
    st.info(f"🔍 Detected Task Type: **{task_type.title()}**")

    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    if task_type == "classification":
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = [
        ("RandomForest", RandomForestClassifier()),
        ("GradientBoosting", GradientBoostingClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("SVM", SVC(probability=True)),
        ("LogisticRegression", LogisticRegression(max_iter=1000))
    ]

    regressors = [
        ("RandomForest", RandomForestRegressor()),
        ("GradientBoosting", GradientBoostingRegressor()),
        ("KNN", KNeighborsRegressor()),
        ("SVM", SVR()),
        ("LinearRegression", LinearRegression())
    ]

    models = classifiers if task_type == "classification" else regressors
    scores = []
    best_models = []

    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds)
        scores.append((name, score))
        best_models.append((name, model))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_models = scores[:3]

    st.subheader("🏆 Top 3 Models")
    for name, score in top_models:
        st.markdown(f"- **{name}** → Score: `{score:.2f}`")

    with PdfPages("model_report.pdf") as pdf:
        plt.figure(figsize=(8, 4))
        names, vals = zip(*scores)
        sns.barplot(x=vals, y=names)
        plt.title("Model Comparison")
        plt.xlabel("Accuracy" if task_type == "classification" else "R2 Score")
        pdf.savefig()
        plt.close()

    st.success("📄 Model Report generated as PDF.")

    recipient_email = st.text_input("Enter Client Email to Send Report")
    
    # Wrap the entire email report sending block in a root trace
    if st.button("📤 Send Visual + ML Reports") and recipient_email:
        
        # Start of the main trace for report sending
        @traceable(run_type="chain", name="Send-AutoML-Reports")
        def send_automl_reports(recipient: str, task_type: str, df: pd.DataFrame):
            
            if not is_valid_email(recipient):
                # Trace for invalid email reply
                auto_msg = EmailMessage()
                auto_msg["Subject"] = "Clarification Needed: Invalid Email"
                auto_msg["From"] = EMAIL_ADDRESS
                auto_msg["To"] = recipient
                auto_msg.set_content("Hi, it seems the email you provided is invalid. Please check and resend.\n\nRegards,\nAgentic AI")
                
                try:
                    _send_smtp_email(auto_msg)
                    st.warning("⚠️ Invalid email address. Auto-replying with clarification message.")
                    return {"status": "clarification_sent", "reason": "invalid_email"}
                except Exception as e:
                    st.error(f"⚠️ Failed to send clarification email: {e}")
                    return {"status": "clarification_failed", "error": str(e)}

            else:
                visual_pdf = generate_visualizations(df)

                # 1. Send Visual Report
                msg1 = EmailMessage()
                msg1["Subject"] = "AutoML Visual Insights Report"
                msg1["From"] = EMAIL_ADDRESS
                msg1["To"] = recipient
                msg1.set_content("Please find attached the data visualizations report.\n\nRegards,\nAgentic AI")
                with open("visual_report.pdf", "rb") as f:
                    msg1.add_attachment(f.read(), maintype="application", subtype="pdf", filename="visual_report.pdf")
                
                visual_result = None
                try:
                    visual_result = _send_smtp_email(msg1)
                except Exception as e:
                    st.error(f"⚠️ Failed to send Visual Report: {e}")
                    # Continue to next step, but log failure

                # 2. Send Model Report
                msg2 = EmailMessage()
                msg2["Subject"] = f"AutoML Report - {task_type.title()}"
                msg2["From"] = EMAIL_ADDRESS
                msg2["To"] = recipient
                msg2.set_content("Attached is the AutoML model performance report.\n\nRegards,\nAgentic AI")
                with open("model_report.pdf", "rb") as f:
                    msg2.add_attachment(f.read(), maintype="application", subtype="pdf", filename="model_report.pdf")
                
                model_result = None
                try:
                    model_result = _send_smtp_email(msg2)
                    st.success("✅ Both Visual & Model Reports sent to your client.")
                    return {"status": "sent_ok", "visual_status": visual_result["status"] if visual_result else "failed", "model_status": model_result["status"] if model_result else "failed"}
                except Exception as e:
                    st.error(f"⚠️ Failed to send Model Report: {e}")
                    return {"status": "partial_failure", "visual_status": visual_result["status"] if visual_result else "failed", "model_error": str(e)}


        # Call the traceable report function
        send_automl_reports(recipient_email, task_type, df)


# === Email Auto-Responder ===
st.markdown("---")
st.header("📬 Auto Email Response (From Gmail Inbox)")

@traceable(run_type="tool") # Tracing the IMAP network call
def fetch_latest_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            return None, None, None

        latest_id = ids[-1]
        result, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        from_email = email_message["From"]
        subject = email_message["Subject"]
        body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()
        
        # Log successful fetch for tracing
        return from_email, subject, body

    except Exception as e:
        # LangSmith will capture this exception and log the run as 'error'
        st.error(f"❌ Error fetching email: {e}")
        raise RuntimeError(f"IMAP fetch failed: {e}") from e


@traceable(run_type="tool") # Reuses the traceable SMTP helper implicitly
def send_auto_reply(to_email, subject, reply_content):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"RE: {subject}"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(reply_content)

        # Use the traceable helper for the actual send operation
        _send_smtp_email(msg)

        return True
    except Exception as e:
        # Error already logged via _send_smtp_email's internal raise
        return False

@traceable(run_type="llm") # Use 'llm' run_type for the call to the foundation model API
def generate_ai_reply(message):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant replying to business emails."},
            {"role": "user", "content": message}
        ]
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses
        content = response.json()["choices"][0]["message"]["content"]
        
        # LangSmith automatically records inputs and outputs, and duration.
        return content
    except Exception as e:
        st.error(f"❌ Failed to generate AI reply: {e}")
        # Re-raise to mark the trace run as failed
        raise RuntimeError(f"LLM API call failed: {e}") from e

if st.button("📥 Check & Auto-Reply to Latest Email"):
    
    # This block itself creates the root trace for the agent run
    @traceable(run_type="agent", name="Auto-Reply-Agent")
    def run_auto_reply_agent():
        from_email, subject, body = fetch_latest_email()
        
        if from_email:
            st.subheader("📨 Incoming Email")
            st.markdown(f"**From:** {from_email}")
            st.markdown(f"**Subject:** {subject}")
            st.text_area("Message", value=body, height=150)

            reply = generate_ai_reply(body)
            st.text_area("🤖 AI Reply", value=reply, height=180)

            if send_auto_reply(from_email, subject, reply):
                st.success("✅ Reply sent successfully.")
                return {"status": "success", "from_email": from_email, "subject": subject}
            else:
                return {"status": "failed_to_send_reply", "from_email": from_email}
        else:
            st.info("📭 No new unread emails.")
            return {"status": "no_new_emails"}

    # Call the traceable agent function
    run_auto_reply_agent()
