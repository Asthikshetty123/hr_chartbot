import streamlit as st
import requests
import json

# Configure page
st.set_page_config(
    page_title="HR Policy Chatbot (Modern Darker Layout)",
    page_icon="📚",
    layout="wide",
    # Consider setting a default theme if you want a system-wide dark mode look
    # theme="dark" # This sets the entire Streamlit app to dark mode
)

# Custom CSS for modern styling and darker layout control
st.markdown("""
<style>
    /* General body font and background if you don't use theme="dark" */
    body {
        color: #e0e0e0; /* Lighter text for darker backgrounds */
        background-color: #2c3e50; /* Darker background */
    }

    /* Streamlit's main block padding adjustment */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #9cdbff; /* A brighter, yet professional blue for the header */
        text-align: center;
        margin-bottom: 1rem;
        padding-top: 1rem;
    }

    /* Message Bubble Styling */
    .message-bubble {
        padding: 12px 18px; /* Slightly larger padding */
        border-radius: 18px; /* More rounded corners */
        margin-bottom: 15px;
        max-width: 75%; /* Slightly less wide to keep focus */
        font-size: 1rem;
        line-height: 1.5;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
    }

    /* Assistant (Left) Bubble - Darker, professional grey/blue */
    .assistant-bubble {
        background-color: #3b506b; /* Darker blue-grey */
        color: #e0e0e0; /* Light text for dark bubble */
        border-top-left-radius: 5px; /* Less rounded on one corner */
    }

    /* User (Right) Bubble - Darker, subtle green/teal */
    .user-bubble {
        background-color: #5c6c60; /* Darker green-grey */
        color: #f0f0f0; /* Light text for dark bubble */
        border-top-right-radius: 5px; /* Less rounded on one corner */
        margin-left: auto; /* Pushes the user bubble to the right */
    }

    /* Overriding Streamlit's default chat message styling to prevent conflicts */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0 !important; /* Remove default padding from the container */
        margin: 0 !important;  /* Remove default margin from the container */
    }
    
    /* Input area styling */
    [data-testid="stChatInput"] {
        padding-top: 1.5rem; /* More space above input */
        padding-bottom: 1rem;
        background-color: #1a222c; /* Darker input background */
        border-top: 1px solid #4a4a4a; /* Separator line */
    }

    /* Text input itself */
    [data-testid="stChatInput"] input {
        background-color: #2c3e50; /* Darker input field */
        color: #f0f0f0; /* Light text in input */
        border: 1px solid #5a5a5a;
        border-radius: 10px;
    }
    [data-testid="stChatInput"] button {
        background-color: #1f77b4; /* Darker send button */
        color: white;
        border-radius: 10px;
    }

    /* Cache status styling */
    .stCaption {
        font-style: italic;
        color: #88c0d0; /* A soft blue/cyan for cache status */
        margin-top: 8px; /* More space above caption */
        font-size: 0.85rem;
        display: block;
    }

    /* Streamlit widgets for overall darker look */
    .stApp {
        background-color: #1a222c; /* Even darker background for the entire app */
    }
</style>
""", unsafe_allow_html=True)

class HRChatbotFrontend:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
    
    def send_query(self, question: str) -> dict:
        """Send query to backend API with robust error handling."""
        try:
            response = requests.post(
                f"{self.backend_url}/query",
                json={"question": question},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"answer": f"**🚫 Error:** Could not connect to backend at `{self.backend_url}`. Details: {str(e)}", "sources": [], "error": True}
    
    def render_message(self, role: str, content: str, from_cache: bool = False):
        """
        Renders a single message using columns for left/right alignment.
        """
        # Use columns to control alignment
        if role == "user":
            # User message: Pushed to the right (empty column on the left)
            # Adjust column width for wider chat area
            col_left, col_right = st.columns([2, 5]) 
            with col_right:
                st.markdown(f'<div class="message-bubble user-bubble">{content}</div>', unsafe_allow_html=True)
        else: # assistant
            # Assistant message: Aligned to the left (empty column on the right)
            # Adjust column width for wider chat area
            col_left, col_right = st.columns([5, 2])
            with col_left:
                message_html = f'<div class="message-bubble assistant-bubble">{content}'
                if from_cache:
                    # Cache status is part of the bubble content for cleaner rendering
                    message_html += '<span class="stCaption">✅ Response served from cache</span>'
                message_html += '</div>'
                st.markdown(message_html, unsafe_allow_html=True)

    def display_chat(self):
        """Display the main chat interface"""
        st.markdown('<div class="main-header">HR Policy Chatbot 🤖</div>', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your HR Policy Advisor. How can I assist you today?", "from_cache": False}
            ]
        
        # Display chat messages using the custom renderer
        for message in st.session_state.messages:
            self.render_message(
                role=message["role"],
                content=message["content"],
                from_cache=message.get("from_cache", False) and message["role"] == "assistant"
            )
        
        # Chat input logic
        if prompt := st.chat_input("Ask about HR policies..."):
            
            # 1. Add and render the user message immediately
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            # Rerun the script to display the user message before the spinner
            st.rerun() 
            
            # The rest of the logic runs on the subsequent rerun after the user message is displayed.

def main():
    chatbot = HRChatbotFrontend()
    chatbot.display_chat()
    
    # Logic to fetch and display the assistant response (runs after user input)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        
        # Check if the last assistant message is a placeholder or not present (to prevent double-querying)
        if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user":
            
            # Get bot response with spinner
            with st.spinner("Searching HR policies..."):
                response = chatbot.send_query(user_prompt)
            
            # Prepare assistant's final message object
            assistant_message = {
                "role": "assistant", 
                "content": response["answer"],
                "from_cache": response.get("from_cache", False)
            }
            
            # Add assistant response to chat history
            st.session_state.messages.append(assistant_message)
            
            # Rerun to display the new assistant message
            st.rerun()

if __name__ == "__main__":
    main()