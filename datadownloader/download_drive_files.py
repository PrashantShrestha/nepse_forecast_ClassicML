import os
from googleapiclient.http import MediaIoBaseDownload
from datadownloader.authenticate import authenticate_google_drive
from datetime import datetime

# Folder ID of your Google Drive folder containing CSVs
FOLDER_ID = '1l3pzHWiS6zdUCX_AXssQsKGhuuo2Xe90'

def should_download_file(service, file_id, local_path):
    """Check if file needs to be downloaded"""
    if not os.path.exists(local_path):
        return True
    
    # Get remote file modified time
    remote_file = service.files().get(
        fileId=file_id, 
        fields="modifiedTime"
    ).execute()
    remote_modified = datetime.strptime(
        remote_file['modifiedTime'], 
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    
    # Get local file modified time
    local_modified = datetime.fromtimestamp(os.path.getmtime(local_path))
    
    return remote_modified > local_modified

def download_csv_files(service, folder_id, download_dir='data/raw'):
    """Download only new/updated CSV files from Google Drive"""
    os.makedirs(download_dir, exist_ok=True)
    print(f"📁 Target directory: {download_dir}")

    # Query for CSV files only
    query = f"'{folder_id}' in parents and mimeType='text/csv'"
    results = service.files().list(
        q=query,
        fields="files(id, name, modifiedTime)"
    ).execute()
    files = results.get('files', [])

    if not files:
        print("⚠️ No CSV files found in the folder.")
        return

    downloaded_count = 0
    skipped_count = 0

    for file in files:
        file_name = file['name']
        file_id = file['id']
        file_path = os.path.join(download_dir, file_name)
        
        if not should_download_file(service, file_id, file_path):
            print(f"⏩ Skipping '{file_name}' (already up-to-date)")
            skipped_count += 1
            continue

        print(f"⏬ Downloading '{file_name}'...")
        try:
            request = service.files().get_media(fileId=file_id)
            with open(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    print(f"   ↳ Progress: {int(status.progress() * 100)}%", end='\r')
            print(f"✅ Saved '{file_name}'")
            downloaded_count += 1
        except Exception as e:
            print(f"❌ Failed to download '{file_name}': {str(e)}")

    print(f"\n📊 Summary: {downloaded_count} new/updated files downloaded, {skipped_count} files skipped")

if __name__ == '__main__':
    try:
        service = authenticate_google_drive()
        download_csv_files(service, FOLDER_ID)
    except Exception as e:
        print(f"🔥 Critical error: {str(e)}")