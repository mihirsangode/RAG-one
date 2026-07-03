from googleapiclient.discovery import build
from google.oauth2 import service_account

# Authenticate using the service account file found in the project
creds_path = r"C:\Users\mihir\Downloads\RAG-one\pdf-rag-500104-bc5304a47272.json"
creds = service_account.Credentials.from_service_account_file(creds_path)
service = build('drive', 'v3', credentials=creds)

# Register the webhook
# NOTE: The address needs to be your actual deployed webhook listener URL
channel = {
    "id": "rag-drive-webhook-channel-1", # A unique ID for this channel
    "type": "web_hook",
    "address": "https://rag-webhook-listener.onrender.com/drive-webhook" 
}

# Use the 'watch' method on the folder ID found in .env
folder_id = "1-1sQtFd_zVRE4H4R-rrOGCZnzYdQQ8Zb"
response = service.files().watch(fileId=folder_id, body=channel).execute()

print("Webhook registered successfully!")
print(response)
