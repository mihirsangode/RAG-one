from fastapi import FastAPI, Request, BackgroundTasks
import subprocess
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

def run_ingest():
    """Executes your existing ingest.py script."""
    logging.info("Webhook received: Starting ingestion.")
    subprocess.run(["python", "ingest.py"])

@app.post("/drive-webhook")
async def receive_google_drive_alert(request: Request, background_tasks: BackgroundTasks):
    # Verify this is a Google Drive change notification
    headers = request.headers
    
    # Google sends headers like 'X-Goog-Resource-State'
    if headers.get("X-Goog-Resource-State") == "update":
        # Add to background tasks so the server responds to Google immediately
        background_tasks.add_task(run_ingest)
        return {"status": "accepted"}
        
    return {"status": "ignored"}