from pathlib import Path


if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)
    print("Ron-1 workspace initialized.")
    print("Run: python model/download_model.py")
    print("Then start the API with: uvicorn api.server:app --host 0.0.0.0 --port 8000")
