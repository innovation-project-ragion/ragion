import requests

# Base URL for the FastAPI service
BASE_URL = "http://localhost:8000/api/v1/documents"

def test_upload_document():
    # Upload a document
    with open("sample_document.txt", "rb") as file:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("Eila 81v SH-4.docx", file, "text/plain")}
        )
        if response.status_code == 200:
            print("Document uploaded successfully.")
            print("Response:", response.json())
        else:
            print("Failed to upload document.")
            print("Error:", response.json())

def test_list_documents():
    # List all documents
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        print("Documents in Milvus:")
        print(response.json())
    else:
        print("Failed to list documents.")
        print("Error:", response.json())

def main():
    print("Starting batch job tests...")
    test_upload_document()
    test_list_documents()
    print("Batch job tests completed.")

if __name__ == "__main__":
    main()
