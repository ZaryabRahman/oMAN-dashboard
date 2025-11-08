import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# --- STATE MANAGEMENT ---
# Initialize session state variables if they don't exist.
if 'last_email_time' not in st.session_state:
    st.session_state.last_email_time = 0

# This state will hold the name of the module that triggered an alert (e.g., 'Fire')
if 'active_alert_module' not in st.session_state:
    st.session_state.active_alert_module = None

# --- CONFIGURATION ---
# Cooldown period in seconds before another EMAIL can be sent.
EMAIL_COOLDOWN = 60  # 1 minute

def trigger_alert(module_name):
    """
    Sends an email if the cooldown has passed and sets the active alert state.
    """
    current_time = time.time()
    
    # Set the UI alert state immediately for visual feedback
    st.session_state.active_alert_module = module_name
    
    # Only send an email if the cooldown period has passed
    if current_time - st.session_state.last_email_time > EMAIL_COOLDOWN:
        st.session_state.last_email_time = current_time  # Reset the timer
        
        try:
            # Load credentials from Streamlit's secrets
            sender_email = st.secrets["SENDER_EMAIL"]
            sender_password = st.secrets["SENDER_PASSWORD"]
            recipient_email = st.secrets["RECIPIENT_EMAIL"]
            
            # Create the email content
            subject = f"🚨 Security Alert: {module_name} Detected!"
            body = (
                f"This is an automated alert from the Intelligent Vision Platform.\n\n"
                f"A potential '{module_name}' event was detected at {time.ctime()}.\n\n"
                "Please review the system footage immediately."
            )
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to the SMTP server (example for Gmail)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            st.toast(f"📧 Email alert for {module_name} sent successfully!")

        except Exception as e:
            st.error(f"Failed to send email: {e}")
            st.warning("Please ensure your `.streamlit/secrets.toml` file is configured correctly.")

def dismiss_alert():
    """Clears the active alert state."""
    st.session_state.active_alert_module = None
