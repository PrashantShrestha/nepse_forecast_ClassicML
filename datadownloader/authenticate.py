from google.oauth2 import service_account
from googleapiclient.discovery import build
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your service account JSON file
SERVICE_ACCOUNT_FILE = os.path.join(SCRIPT_DIR, 'service-account.json')

# Scopes for the Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Authenticate with Google using the service account
def authenticate_google_drive():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

# Example usage
if __name__ == '__main__':
    service = authenticate_google_drive()
    print("Authenticated to Google Drive.")
