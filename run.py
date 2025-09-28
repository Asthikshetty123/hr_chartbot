import subprocess
import sys
import time
import threading
from backend import app
import uvicorn

def run_backend():
    """Run FastAPI backend"""
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

def run_frontend():
    """Run Streamlit frontend"""
    # Wait for backend to start
    time.sleep(3)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"])

if __name__ == "__main__":
    print("Starting HR RAG Chatbot...")
    print("Backend will run on http://localhost:8000")
    print("Frontend will run on http://localhost:8501")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Start frontend
    run_frontend()