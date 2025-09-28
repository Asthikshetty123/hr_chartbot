import os
import requests

def setup_environment():
    """Setup environment and download sample HR policy if needed"""
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    
    # Check if Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY environment variable")
        print("You can get a free API key from: https://console.groq.com/")
        api_key = input("Enter your Groq API key: ")
        with open(".env", "w") as f:
            f.write(f"GROQ_API_KEY={api_key}\n")
        print("API key saved to .env file")
    
    # Download sample HR policy if no PDF exists
    if not os.path.exists("hr_policy.pdf"):
        print("No HR policy PDF found. Please place your HR policy PDF as 'hr_policy.pdf' in the root directory.")
        # Alternatively, you could provide a sample download here

if __name__ == "__main__":
    setup_environment()