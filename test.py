import os
import json
import requests
from typing import Dict, Any, List, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.tools import BaseTool

load_dotenv()

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


# Run a test for `_transform_patient_data` only
if __name__ == "__main__":
    # Mock Validation Result (Simulating a Validated Patient Record)
    validation_result = {
        "valid": True,  # Ensure it's marked as valid
        "missing_fields": [],
        "to_email": "john.doe@example.com",
        "original_subject": "Prescription Request",
        "original_message_id": "123456",
        "extracted_data": {
            "patient_details": {
                "name": "John Doe",
                "address": "123 Main St",
                "email": "john.doe@example.com"
            },
            "pharmacy": {
                "name": "ABC Pharmacy",
                "address": "456 Elm St"
            },
            "vital_signs": {
                "blood_pressure": "120/80",
                "heart_rate": 72,
                "date_of_readings": "2025-02-20"
            },
            "medications": [
                {
                    "drug_name": "Lisinopril",
                    "strength": "10mg",
                    "repeats": 2
                },
                {
                    "drug_name": "Atorvastatin",
                    "strength": "20mg",
                    "repeats": 1
                },
                {
                    "drug_name": "dolo",
                    "strength": "30mg",
                    "repeats": 4
                },
                {
                    "drug_name": "paracitamol",
                    "strength": "90mg",
                    "repeats": 2
                }
            ]
        }
    }

    # Prepare data for transformation
    patient_data = {
        "email": validation_result["to_email"],
        "subject": validation_result["original_subject"],
        "message_id": validation_result["original_message_id"],
        "extracted_data": validation_result["extracted_data"]
    }


    # Instantiate the tool
    upload_tool = PatientDataUploadTool()

    # Run the tool
    response = upload_tool._run(validation_result)

    # Print the response
    print("API Response:", response)


