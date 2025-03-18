import os
import json
import re
import smtplib
import email
import imaplib
import requests
import pickle
from hashlib import sha256
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.header import decode_header
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from crewai.tools import BaseTool
from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field
# import groq
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# Load environment variables
load_dotenv()
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("EMAIL_HOST")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
VERODAT_AI_API_KEY = os.getenv("VERODAT_AI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = groq.Client(api_key=GROQ_API_KEY)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(BASE_DIR, "Full_patient_data.json")
PICKLE_FILE = os.path.join(BASE_DIR, "validation_records.pkl")

def connect_imap():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        return mail, "OK"
    except Exception as e:
        print(f"IMAP Connection Error: {e}")
        return None, f"IMAP Connection Error: {e}"

def close_imap(mail):
    if mail:
        mail.logout()

class FetchEmailInput(BaseModel):
    pass  

class FetchEmailTool(BaseTool):
    name: str = "fetch_email"
    description: str = "Fetch unread emails related to prescriptions, replies, or without a subject."
    args_schema: Type[BaseModel] = FetchEmailInput

    def _run(self):
        #  Internal list of prescription-related keywords
        prescription_keywords = ["prescription", "medication", "rx", "medicine", "pharmacy", "treatment"]

        mail, status = connect_imap()
        if not mail:
            print("❌ Failed to connect to IMAP.")
            return []

        mail.select("inbox")
        status, messages = mail.search(None, "UNSEEN")

        if status != "OK" or not messages[0] or messages[0] == b'':
            print("📭 No unread emails found.")
            close_imap(mail)
            return []

        emails = []
        print(f"🔍 Unread Emails Found: {len(messages[0].split())}")

        for num in messages[0].split():
            print(f"📥 Fetching email ID: {num.decode()}")
            _, msg_data = mail.fetch(num, "(RFC822)")

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    sender = msg["From"]
                    subject = msg["Subject"]
                    message_id = msg["Message-ID"]
                    email_type = "reply" if msg["In-Reply-To"] else "fresh"

                    #  Decode subject if encoded
                    if subject:
                        decoded_subject, encoding = decode_header(subject)[0]
                        subject = decoded_subject.decode(encoding) if isinstance(decoded_subject, bytes) else decoded_subject
                    else:
                        subject = ""

                    print(f"📨 Extracted Email - Sender: {sender}, Subject: '{subject}', Message ID: {message_id}, Type: {email_type}")

                    #  Check if subject is related to prescriptions, replies (Re:), or is empty
                    if (
                        not subject
                        or subject.lower().startswith("re:")
                        or any(keyword.lower() in subject.lower() for keyword in prescription_keywords)
                    ):
                        emails.append({
                            "sender": sender,
                            "subject": subject,
                            "message_id": message_id,
                            "type": email_type
                        })
                    else:
                        print("❌ Email does not match criteria. Ignoring.")

        close_imap(mail)
        print(f"📩 Returning {len(emails)} filtered emails.")
        return emails if emails else []


class ExtractFreshBodyInput(BaseModel):
    message_id: str = Field(..., description="Message ID of the email to extract body from.")

class ExtractFreshBodyTool(BaseTool):
    name: str = "extract_fresh_body"
    description: str = "Extract the body of a fresh email using its Message-ID."
    args_schema: Type[BaseModel] = ExtractFreshBodyInput

    def _run(self, message_id: str):
        print(f"ExtractFreshBodyTool called with message_id: {message_id}")

        mail, status = connect_imap()
        if not mail:
            print("Failed to connect to IMAP.")
            return {"body": "", "sender": "", "subject": "", "message_id": message_id}
        
        mail.select("inbox")

        print(f"Searching for email with Message-ID: {message_id}")
        status, messages = mail.search(None, f'HEADER Message-ID "{message_id}"')

        print(f"Search Status: {status}, Messages: {messages}")

        if status != "OK" or not messages[0]:
            print("Email with the given Message-ID not found.")
            close_imap(mail)
            return {"body": "", "sender": "", "subject": "", "message_id": message_id}
        
        for num in messages[0].split():
            print(f"Fetching email ID: {num.decode()}")
            _, msg_data = mail.fetch(num, "(RFC822)")

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    print("Processing email content...")
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Extract sender and subject
                    sender = msg["From"]
                    subject = msg["Subject"]

                    body = ""
                    if msg.is_multipart():
                        print("Email is multipart. Extracting text parts...")
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            print(f"Found content type: {content_type}")

                            if content_type == "text/plain":
                                extracted_text = part.get_payload(decode=True).decode(errors="ignore")
                                body += extracted_text
                                print(f"Extracted text part: {extracted_text[:100]}...")  # Print first 100 chars
                    else:
                        print("Email is not multipart. Extracting body directly...")
                        body = msg.get_payload(decode=True).decode(errors="ignore")
                        print(f"Extracted body: {body[:100]}...")  # Print first 100 chars

                    close_imap(mail)
                    print("Returning extracted email body with sender, subject, and message_id.")
                    return {
                        "body": body.strip(),
                        "sender": sender,
                        "subject": subject,
                        "message_id": message_id
                    }
        
        print("No email body extracted.")
        close_imap(mail)
        return {"body": "", "sender": "", "subject": "", "message_id": message_id}


def clean_email_body(body):
    """Removes quoted text (previous replies) and cleans up content."""
    lines = body.split("\n")
    clean_lines = []
    for line in lines:
        if line.strip().startswith(">") or "On" in line and "wrote:" in line:
            break  # Stop at quoted text
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()

def get_email_body(msg):
    """Extracts and cleans the email body while detecting type (HTML or plain text)."""
    body = None
    email_type = "Unknown"

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode(errors="ignore")
                email_type = "Plain Text"
                break
            elif content_type == "text/html" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode(errors="ignore")
                email_type = "HTML"
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")
        email_type = msg.get_content_type()

    if body:
        if email_type == "HTML":
            body = BeautifulSoup(body, "html.parser").get_text()  # Convert HTML to plain text
        body = clean_email_body(body)  # Remove quoted text

    return body if body else "(No content)"

def fetch_email_by_id(mail, message_id):
    """Fetches an email by its Message-ID."""
    status, messages = mail.search(None, f'HEADER Message-ID "{message_id}"')
    if status == "OK" and messages[0]:
        msg_num = messages[0].split()[0]
        _, msg_data = mail.fetch(msg_num, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                return email.message_from_bytes(response_part[1])
    return None

def extract_all_message_ids(mail, start_message_id):
    """Recursively extracts all message IDs in a thread while maintaining order."""
    message_ids_list = []  
    message_queue = [start_message_id]
    queued_ids = set(message_queue)

    while message_queue:
        message_id = message_queue.pop(0)
        if message_id in message_ids_list:
            continue  

        message_ids_list.append(message_id)
        msg = fetch_email_by_id(mail, message_id)

        if not msg:
            continue 

        # Extract References (Get previous message IDs)
        references = msg["References"]
        in_reply_to = msg["In-Reply-To"]

        if references:
            ref_ids = references.split()  
            for ref_id in ref_ids:
                if ref_id not in message_ids_list and ref_id not in queued_ids:
                    message_queue.append(ref_id)
                    queued_ids.add(ref_id)

        if in_reply_to and in_reply_to not in message_ids_list and in_reply_to not in queued_ids:
            message_queue.append(in_reply_to)
            queued_ids.add(in_reply_to)

    return message_ids_list  # Return ordered message IDs


class FetchThreadBodyToolInput(BaseModel):
    message_id: str = Field(..., description="Message ID of the email thread to fetch")

class FetchThreadBodyTool(BaseTool):
    name: str = "fetch_thread_body"
    description: str = "Fetches the full email thread given a message_id."
    args_schema: Type[BaseModel] = FetchThreadBodyToolInput

    def _run(self, message_id: str) -> Dict:
        print(f"FetchThreadBodyTool called with message_id: {message_id}")

        mail, status = connect_imap()
        if not mail:
            print("IMAP connection failed.")
            return {
                "thread_body": "IMAP connection failed.",
                "sender": "",
                "subject": "",
                "message_id": message_id
            }
        
        mail.select("inbox")

        # First get the sender and subject of the original message
        original_sender = ""
        original_subject = ""
        
        print(f"Fetching original email with Message-ID: {message_id}")
        original_msg = fetch_email_by_id(mail, message_id)
        if original_msg:
            original_sender = original_msg["From"]
            original_subject = original_msg["Subject"]
            print(f"Original email - Sender: {original_sender}, Subject: {original_subject}")

        print(f"Extracting thread messages for Message-ID: {message_id}")
        thread_message_ids = extract_all_message_ids(mail, message_id)
        print(f"Extracted thread message IDs: {thread_message_ids}")

        full_thread_text = ""
        
        for msg_id in thread_message_ids:
            print(f"Fetching email with Message-ID: {msg_id}")
            msg = fetch_email_by_id(mail, msg_id)

            if not msg:
                print(f"Failed to fetch email for Message-ID: {msg_id}")
                continue

            sender = msg["From"]
            print(f"Email sender: {sender}")

            if EMAIL_USERNAME in sender:
                print("Skipping own email.")
                continue

            body_text = get_email_body(msg)

            full_thread_text += f"\n\n{body_text}"
        
        close_imap(mail)
        return {
            "thread_body": full_thread_text.strip(),
            "sender": original_sender,
            "subject": original_subject,
            "message_id": message_id
        }
    
def extract_json_from_text(text):
    """Extracts JSON content from a mixed text response."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None 

def clean_extracted_data(data):
    """Ensure missing fields are set to None instead of empty strings."""
    if isinstance(data, dict):
        return {k: clean_extracted_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_extracted_data(v) for v in data]
    elif data in ["", "N/A", "None", "null"]:
        return None
    return data


class ExtractPatientDetailsInput(BaseModel):
    email_body: str = Field(..., description="Email body content to extract patient details from.")
    sender: str = Field("", description="Email sender")
    subject: str = Field("", description="Email subject")
    message_id: str = Field("", description="Email message ID")

class ExtractPatientDetailsTool(BaseTool):
    name: str = "extract_patient_details"
    description: str = "Extract structured patient details from an email thread using LLM."
    args_schema: Type[BaseModel] = ExtractPatientDetailsInput

    def _run(self, email_body: str, sender="", subject="", message_id=""):

        prompt = f"""
        **Task:**  
        Extract structured medical details from the email below.  

        **Rules:**  
        - Extract **only explicitly mentioned** details from the provided email.  
        - If any field is **missing**, return `null`.  
        - **Do NOT assume or infer** missing details from context or previous emails.  
        - Ensure numbers (e.g., heart rate, repeats) are **integers**.  
        - **Do NOT copy or take details from the example email.**  
        - **Return output strictly in JSON format.** No extra text, explanations, or assumptions.
        - **Return exact output as below format.** not put emply list if any catagory is missing

        **Expected JSON Output Format:**  
        ```json
        {{
            "patient_details": {{
                "name": "Full Name (or null if missing)",
                "address": "Home Address (or null)",
                "email": "Email Address (or null)"
            }},
            "pharmacy": {{
                "name": "Pharmacy Name (or null)",
                "address": "Pharmacy Address (or null)"
            }},
            "vital_signs": {{
                "blood_pressure": "e.g., 120/80 (or null)",
                "heart_rate": null,
                "date_of_readings": "YYYY-MM-DD (or null)"
            }},
            "medications": {{
                "drug_name": null,
                "strength": null,
                "repeats": null
            }}
        }}
        ```

        **Email Content (Extract Data From This Only):**  
        ```  
        {email_body}  
        ```  
        """
        try:
            response = client.chat.completions.create(
                model="openai/chatgpt-4o-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            # response = client.chat.completions.create(
            #     model="llama3-8b-8192",
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.1,
            #     max_tokens=500
            # )
            extracted_text = response.choices[0].message.content.strip()
            json_str = extract_json_from_text(extracted_text)

            if json_str:
                extracted_data = json.loads(json_str)

                extracted_data = clean_extracted_data(extracted_data)
                print("[ExtractPatientDetailsTool] Cleaned extracted data:", extracted_data)
            else:
                print("[ExtractPatientDetailsTool] No valid JSON extracted.")
                extracted_data = {}

        except json.JSONDecodeError as e:
            print(f"[ExtractPatientDetailsTool] JSONDecodeError: {e}")
            extracted_data = {}

        except Exception as e:
            print(f"[ExtractPatientDetailsTool] Unexpected Error: {e}")
            extracted_data = {}

        # Add email metadata to the return
        result = {
            "extracted_data": extracted_data,
            "sender": sender,
            "subject": subject,
            "message_id": message_id
        }
        
        print("[ExtractPatientDetailsTool] Returning extracted data with metadata:", result)
        return result

class ValidateDataToolInput(BaseModel):
    extracted_data: Dict[str, Any] = Field(..., description="Extracted patient data to validate")
    sender: str = Field(..., description="Email sender (explicitly provided)")
    subject: str = Field(..., description="Email subject (explicitly provided)")
    message_id: str = Field(..., description="Email message ID (explicitly provided)")

class ValidateDataTool(BaseTool):
    name: str = "validate_data"
    description: str = "Validates extracted patient data and stores validation records in pickle."
    args_schema = ValidateDataToolInput

    def _run(self, extracted_data: Dict[str, Any], sender: str, subject: str, message_id: str) -> List[Dict[str, Any]]:
        """Runs validation on extracted patient data"""

        # Check if data follows new format (has 'email_metadata' key)
        if "email_metadata" in extracted_data:
            metadata = extracted_data["email_metadata"]
            sender = metadata.get("sender", sender)
            subject = metadata.get("subject", subject)
            message_id = metadata.get("message_id", message_id)

        print(f"📤 Processing Email - Sender: {sender}, Subject: '{subject}', Message ID: {message_id}")

        required_fields = {
            "patient_details": ["name", "address", "email"],
            "pharmacy": ["name", "address"],
            "vital_signs": ["blood_pressure", "heart_rate", "date_of_readings"],
            "medications": ["drug_name", "strength", "repeats"]  # Handling list format
        }

        missing_fields = []

        # Validate standard dictionary fields
        for category, fields in required_fields.items():
            if category in extracted_data:
                if isinstance(extracted_data[category], list):  # Handle list-based categories
                    for entry in extracted_data[category]:  # Iterate over list of meds
                        for field in fields:
                            if not entry.get(field) or str(entry[field]).strip() == "":
                                missing_fields.append(f"{category}.{field}")
                else:  # Handle dictionary-based categories
                    for field in fields:
                        if not extracted_data[category].get(field) or str(extracted_data[category][field]).strip() == "":
                            missing_fields.append(f"{category}.{field}")
            else:
                # Entire category is missing
                missing_fields.extend([f"{category}.{field}" for field in fields])

        validation_result = {
            "valid": not missing_fields,
            "missing_fields": missing_fields,
            "to_email": sender,
            "original_subject": subject,
            "original_message_id": message_id,
            "extracted_data": extracted_data  # ✅ Added extracted data to the result
        }

        print(f"🔎 Validation Result: {validation_result}")

        # Normalize email format
        normalized_email = self.normalize_email(sender)

        # Generate prescription ID
        patient_data = extracted_data.get("patient_details", {})
        patient_name = str(patient_data.get("name", "")).strip()
        patient_email = str(patient_data.get("email", "")).strip()

        if not patient_name or not patient_email:
            print("⚠️ Warning: Missing patient details (name/email). Prescription ID may be unreliable!")

        prescription_id = self.generate_prescription_id(patient_name, patient_email)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status = "Pending" if missing_fields else "Done"

        self.save_or_update_record(normalized_email, prescription_id, message_id, current_time, status, missing_fields)

        if not missing_fields:
            self.save_to_json(extracted_data, patient_email, subject, message_id)

        return validation_result

    def generate_prescription_id(self, patient_name, patient_email):
        """Generates a unique prescription ID based on patient name + email."""
        if not patient_name or not patient_email:
            print("⚠️ Warning: Prescription ID may be unreliable due to missing patient information!")
            return "unknown_id"

        unique_string = f"{patient_name}-{patient_email}"
        return sha256(unique_string.encode()).hexdigest()[:10] 

    def normalize_email(self, email):
        """Extract only the email address from 'Name <email@example.com>' format."""
        match = re.search(r"<(.+?)>", email)
        return match.group(1) if match else email.strip()

    def save_or_update_record(self, email, prescription_id, message_id, validation_time, status, missing_fields):
        """Updates existing record or adds a new one while storing only required metadata."""
        records = self.load_from_pickle()
        record_found = False

        for record in records:
            if record["email"] == email and record["prescription_id"] == prescription_id:
                if not missing_fields:
                    print(f"🔄 Updating existing record for Prescription ID: {prescription_id}")
                    record.update({
                        "status": "Done",
                        "validation_time": validation_time,
                        "missing_fields": [],
                        "latest_message_id": message_id
                    })
                else:
                    print(f"🔍 Existing record found but still has missing fields: {missing_fields}")
                record_found = True
                break

        if not record_found:
            records.append({
                "email": email,
                "prescription_id": prescription_id,
                "latest_message_id": message_id,
                "validation_time": validation_time,
                "status": status,
                "missing_fields": missing_fields,
                "last_reminder_time": None
            })
            print(f"➕ Added new prescription for Email: {email}, Prescription ID: {prescription_id}")

        self.save_to_pickle(records)

    def save_to_json(self, extracted_data, patient_email, subject, message_id):
        """Saves fully validated extracted data to a JSON file without overwriting existing records."""
        json_data = {
            "email": patient_email,
            "subject": subject,
            "message_id": message_id,
            "extracted_data": extracted_data
        }

        existing_data = []
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    print("⚠️ JSON file is corrupted. Resetting it...")
                    existing_data = []

        existing_data.append(json_data)

        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

        print(f" Extracted data saved to JSON: {JSON_FILE}")

    def save_to_pickle(self, records):
        """Safely saves updated validation records to the pickle file."""
        try:
            with open(PICKLE_FILE, "wb") as f:
                pickle.dump(records, f)
            print(" Validation records updated successfully!")
        except Exception as e:
            print(f"❌ Error saving to pickle: {e}")

    def load_from_pickle(self):
        """Loads existing validation records from the pickle file safely."""
        if not os.path.exists(PICKLE_FILE):
            print("⚠️ Pickle file does not exist. Creating a new one...")
            self.save_to_pickle([])
            return []

        try:
            with open(PICKLE_FILE, "rb") as f:
                records = pickle.load(f)
                print(f"📂 Loaded {len(records)} records from pickle file.")
                return records
        except (EOFError, pickle.UnpicklingError):
            print("⚠️ Pickle file is corrupted. Resetting it...")
            self.save_to_pickle([])
            return []

class SendReplyEmailInput(BaseModel):
    to_email: str = Field(..., description="Recipient email address.")
    original_subject: str = Field(..., description="Original email subject.")
    original_message_id: str = Field(..., description="Original message ID for threading.")
    missing_fields: list = Field(..., description="List of missing details to request.")

class SendReplyEmailTool(BaseTool):
    name: str = "send_reply_email"
    description: str = "Send a reply email requesting missing details."
    args_schema: Type[BaseModel] = SendReplyEmailInput

    def _run(self, to_email: str, original_subject: str, original_message_id: str, missing_fields: list):
        print(f"📩 Recipient: {to_email}")
        print(f"📜 Original Subject: {original_subject}")
        print(f"🆔 Original Message ID: {original_message_id}")
        print(f"❓ Missing Fields: {missing_fields}")

        subject = f"Re: {original_subject.strip()}" if original_subject.strip() else ""
        print(f"✏️ Computed Subject: {subject}")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"Clinic <{EMAIL_USERNAME}>"
        msg["To"] = to_email
        msg["Reply-To"] = EMAIL_USERNAME

        if original_message_id:
            msg["In-Reply-To"] = original_message_id
            msg["References"] = original_message_id
            print("🔗 Threading information added.")

        body = "Dear Patient,\n\nWe noticed that your request is missing some details. Please provide:\n"
        body += "\n".join(f"- {field}" for field in missing_fields) + "\n\nBest regards,\nClinic"
        msg.set_content(body)

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(" Reply sent successfully.")
            return "Reply sent successfully."
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return f"Error: {e}"
        
class SendConfirmationEmailInput(BaseModel):
    to_email: str = Field(..., description="Recipient email address.")
    original_subject: str = Field(..., description="Original email subject.")
    original_message_id: str = Field(..., description="Original message ID for threading.")

class SendConfirmationEmailTool(BaseTool):
    name: str = "send_confirmation_email"
    description: str = "Send a confirmation email when all required details are provided."
    args_schema: Type[BaseModel] = SendConfirmationEmailInput

    def _run(self, to_email: str, original_subject: str, original_message_id: str):
        print(f"📩 Recipient: {to_email}")
        print(f"📜 Original Subject: {original_subject}")
        print(f"🆔 Original Message ID: {original_message_id}")

        subject = f"{original_subject.strip()}" if original_subject.strip() else ""
        print(f"✏️ Computed Subject: {subject}")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"Clinic <{EMAIL_USERNAME}>"
        msg["To"] = to_email
        msg["Reply-To"] = EMAIL_USERNAME

        if original_message_id:
            msg["In-Reply-To"] = original_message_id
            msg["References"] = original_message_id
            print("🔗 Threading information added.")

        body = "Dear Patient,\n\nThank you for providing all the necessary information. Your request has been received and is being processed.\n\n"
        body += "We will get back to you shortly with further details.\n\nBest regards,\nClinic"
        
        msg.set_content(body)
        print("✍️ Email body prepared.")
        print(f"📨 Email Preview:\n{body}")

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(" Confirmation email sent successfully.")
            return "Confirmation email sent successfully."
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return f"Error: {e}"


class SendReminderInput(BaseModel):
    pass

class SendReminderAndEscalationTool(BaseTool):
    name: str = "send_reminder_and_escalation"
    description: str = "Checks pending validations and sends reminder or escalation emails as needed, but only once per stage."
    args_schema: Type[BaseModel] = SendReminderInput 

    def _run(self):

        if not os.path.exists(PICKLE_FILE):
            print("⚠️ No validation records found.")
            return {"message": "No validation records found."}

        with open(PICKLE_FILE, "rb") as f:
            try:
                records = pickle.load(f)
                print(f"📂 Loaded {len(records)} records from pickle file.")
            except EOFError:
                print("⚠️ Pickle file is empty or corrupted.")
                return {"message": "Pickle file is empty or corrupted."}

        now = datetime.now()
        emails_sent = []

        for record in records:
            if record.get("status") == "Pending":
                validation_time = datetime.strptime(record["validation_time"], "%Y-%m-%d %H:%M:%S")
                time_since_validation = now - validation_time

                reminder_sent = record.get("reminder_sent", False)
                escalation_sent = record.get("escalation_sent", False)

                if time_since_validation >= timedelta(minutes=2) and not reminder_sent:
                    email_type = "reminder"
                    missing_fields = record.get("missing_fields", [])
                    self.send_email(
                        email_type,
                        record["email"],
                        record.get("original_subject", ""),
                        record.get("latest_message_id", ""),
                        missing_fields
                    )
                    record["reminder_sent"] = True  
                    emails_sent.append({"email": record["email"], "type": email_type})

                elif time_since_validation >= timedelta(minutes=5) and not escalation_sent:
                    email_type = "escalation"
                    missing_fields = record.get("missing_fields", [])
                    self.send_email(
                        email_type,
                        record["email"],
                        record.get("original_subject", ""),
                        record.get("latest_message_id", ""),
                        missing_fields
                    )
                    record["escalation_sent"] = True 
                    emails_sent.append({"email": record["email"], "type": email_type})

        self.save_to_pickle(records)

        return {"emails_sent": emails_sent}

    def send_email(self, email_type, recipient_email, original_subject, original_message_id, missing_fields):
        """Sends a reminder or escalation email, but only once per stage."""
        subject_prefix = "" if email_type == "reminder" else "No Response - "
        subject = f"{subject_prefix}{original_subject.strip()}"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"Clinic <{EMAIL_USERNAME}>"
        msg["To"] = recipient_email if email_type == "reminder" else EMAIL_USERNAME  #  Send escalation emails to you
        msg["Reply-To"] = EMAIL_USERNAME

        if original_message_id:
            msg["In-Reply-To"] = original_message_id
            msg["References"] = original_message_id
            print("🔗 Threading information added.")

        #  Format missing fields for the email body
        if missing_fields:
            missing_fields_text = "\n".join([f"- {field}" for field in missing_fields])
        else:
            missing_fields_text = "None  All required fields are filled."

        if email_type == "reminder":
            body = f"""Dear Patient,

We noticed that we haven't received the required information from you yet. 
Please provide the missing details as soon as possible so we can proceed with your prescription request.

📌 **Missing Fields:**  
{missing_fields_text}

If you have already responded, please ignore this email.

Best regards,  
Clinic"""
        else:  # Escalation
            body = f"""Dear {EMAIL_USERNAME},

The patient {recipient_email} has not responded for more than 96 hours regarding their prescription request.
This is a test escalation email sent to you.

Best regards,  
Automated Reminder System"""

        msg.set_content(body)
        print(f"📧 Preparing to send {email_type} email...")
        print(f"📩 Recipient: {recipient_email if email_type == 'reminder' else EMAIL_USERNAME}")
        print(f"📜 Original Subject: {original_subject}")
        print(f"🆔 Original Message ID: {original_message_id}")
        print(f"📨 Email Preview:\n{body}")

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f" {email_type.capitalize()} email sent successfully.")
        except Exception as e:
            print(f"❌ Error sending {email_type} email: {e}")

    def save_to_pickle(self, records):
        """Safely saves updated validation records to the pickle file."""
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(records, f)
        print(" Validation records updated successfully!")

class PatientDataInput(BaseModel):
    validation_result: Dict[str, Any] = Field(..., description="Validation result containing extracted data and validation details")

class PatientDataUploadTool(BaseTool):
    name: str = "upload_patient_data"
    description: str = "Upload validated patient data to a specific dataset in Verodat."
    args_schema: Type[BaseModel] = PatientDataInput

    def _run(self, validation_result: Dict[str, Any]) -> str:
        """
        Transform and upload validated patient data to Verodat **only if required fields are present**.

        Args:
            validation_result: The validation result containing extracted data and metadata.

        Returns:
            Response from the API.
        """
        extracted_data = validation_result.get("extracted_data", {})
        email = validation_result.get("to_email", "")
        subject = validation_result.get("original_subject", "")
        message_id = validation_result.get("original_message_id", "")
        missing_fields = validation_result.get("missing_fields", [])

        # ✅ Check if all required fields are present (missing_fields should be empty)
        if missing_fields:
            return f"❌ Validation failed. Missing fields: {', '.join(missing_fields)}. Data not uploaded."

        # Fetch environment variables
        api_key = os.getenv("VERODAT_AI_API_KEY")
        base_url = os.getenv("VERODAT_BASE_URL", "https://dev-app.verodat.io/api/v3")
        mcp_api_url = os.getenv("VERODAT_MCP_URL", "https://dev-mcp.verodat.io/api/prompt")
        account_id = os.getenv("VERODAT_ACCOUNT_ID")
        workspace_id = os.getenv("VERODAT_WORKSPACE_ID")
        dataset_id = os.getenv("VERODAT_DATASET_ID")

        if not all([api_key, account_id, workspace_id, dataset_id]):
            return "❌ Missing required environment variables. Upload aborted."

        # Prepare patient data
        patient_data = {
            "email": email,
            "subject": subject,
            "message_id": message_id,
            "extracted_data": extracted_data
        }

        # Transform the data
        transformed_data = self._transform_patient_data(patient_data)

        print(f"📤 Uploading validated patient data to dataset {dataset_id}")

        headers = {
            "ai-api-key": api_key,
            "api-base-url": base_url,
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": f"upload sample data into dataset {dataset_id} in workspace {workspace_id} of account {account_id}. {transformed_data}"
        }

        try:
            response = requests.post(mcp_api_url, headers=headers, json=payload, timeout=300)

            if response.status_code in [200, 201]:
                print(f"✅ Patient data successfully uploaded to dataset {dataset_id}")
                return response.text
            else:
                return f"❌ Error {response.status_code}: {response.text}"

        except requests.exceptions.RequestException as e:
            return f"❌ Request failed: {e}"

    def _transform_patient_data(self, patient_data):
        """Transform nested patient data into a separate row for each medication and return a single-line JSON."""
        email = patient_data.get("email", "")
        subject = patient_data.get("subject", "")
        message_id = patient_data.get("message_id", "")

        patient_details = patient_data.get("extracted_data", {}).get("patient_details", {})
        pharmacy = patient_data.get("extracted_data", {}).get("pharmacy", {})
        vital_signs = patient_data.get("extracted_data", {}).get("vital_signs", {})
        medications = patient_data.get("extracted_data", {}).get("medications", [])

        # Ensure medications is always a list
        if not isinstance(medications, list):
            medications = [medications]

        flattened_data = []
        for med in medications:
            flattened_data.append({
                "Sender_Email": email,
                "Subject": subject,
                "Message_ID": message_id,
                "Patient_Name": patient_details.get("name", ""),
                "Patient_Address": patient_details.get("address", ""),
                "Patient_Email": patient_details.get("email", ""),
                "Pharmacy_Name": pharmacy.get("name", ""),
                "Pharmacy_Address": pharmacy.get("address", ""),
                "Blood_Pressure": vital_signs.get("blood_pressure", ""),
                "Heart_Rate": str(vital_signs.get("heart_rate", "")),
                "Reading_Date": vital_signs.get("date_of_readings", ""),
                "Medication": med.get("drug_name", ""),
                "Medication_Strength": med.get("strength", ""),
                "Medication_Repeats": str(med.get("repeats", ""))
            })
        single_line_json = json.dumps({"output": flattened_data}, separators=(',', ':'))

        return single_line_json

if __name__ == "__main__":
    print("🚀 Starting Email Processing Workflow...")

    # Step 1: Fetch unread emails
    fetch_email_tool = FetchEmailTool()
    unread_emails = fetch_email_tool._run()

    if not unread_emails:
        print("📭 No unread prescription-related emails found.")
    else:
        for email_data in unread_emails:
            print(f"\n📩 Processing Email: {email_data['message_id']}")

            # Step 2: Extract body of the fresh email
            extract_body_tool = ExtractFreshBodyTool()
            email_body_data = extract_body_tool._run(email_data["message_id"])

            if not email_body_data["body"]:
                print(f"⚠️ No body extracted for email ID: {email_data['message_id']}")
                continue

            # Step 3: Extract patient details using LLM
            extract_patient_tool = ExtractPatientDetailsTool()
            patient_details = extract_patient_tool._run(
                email_body=email_body_data["body"],
                sender=email_body_data["sender"],
                subject=email_body_data["subject"],
                message_id=email_body_data["message_id"]
            )

            if not patient_details["extracted_data"]:
                print(f"⚠️ No structured data extracted for email ID: {email_data['message_id']}")
                continue

            # Step 4: Validate extracted patient data
            validate_tool = ValidateDataTool()
            validation_result = validate_tool._run(
                extracted_data=patient_details["extracted_data"],
                sender=email_body_data["sender"],
                subject=email_body_data["subject"],
                message_id=email_body_data["message_id"]
            )

            print(f"🔎 Validation Result: {validation_result}")

            # Step 5: Take action based on validation result
            if validation_result["valid"]:
                # If all required fields are present, upload patient data
                print("✅ All required fields are present. Uploading data to Verodat...")
                upload_tool = PatientDataUploadTool()
                upload_response = upload_tool._run(validation_result)
                print(f"📤 Upload Response: {upload_response}")

                # Send confirmation email
                send_confirmation_tool = SendConfirmationEmailTool()
                confirmation_response = send_confirmation_tool._run(
                    to_email=validation_result["to_email"],
                    original_subject=validation_result["original_subject"],
                    original_message_id=validation_result["original_message_id"]
                )
                print(f"📩 Confirmation Email Sent: {confirmation_response}")

            else:
                # If required fields are missing, send a request email for missing details
                print("❌ Missing required fields. Sending request for missing details...")
                send_reply_tool = SendReplyEmailTool()
                reply_response = send_reply_tool._run(
                    to_email=validation_result["to_email"],
                    original_subject=validation_result["original_subject"],
                    original_message_id=validation_result["original_message_id"],
                    missing_fields=validation_result["missing_fields"]
                )
                print(f"📩 Reply Sent: {reply_response}")

    # Step 6: Check for pending validation records and send reminders or escalations if necessary
    print("\n⏳ Checking for pending validations and sending reminders if needed...")
    reminder_tool = SendReminderAndEscalationTool()
    reminder_response = reminder_tool._run()
    print(f"📨 Reminder & Escalation Status: {reminder_response}")

    print("\n🎉 Email processing workflow completed!")