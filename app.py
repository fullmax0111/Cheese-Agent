import streamlit as st
import os
import sys
import json
# Add the project root to the Python path to allow imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from the project
from src.graph.workflow import create_workflow
from langgraph.types import Command

# Page configuration
st.set_page_config(
    page_title="Cheese Agent",
    page_icon="🧀",
    layout="wide"
)

# Function to process workflow response
def process_workflow_response(state):
    # Extract relevant information from the state
    print(state)
    if state.get("reasoning_steps"):
        st.session_state.reasoning_steps.extend(state["reasoning_steps"])
    
    # Check if human-in-the-loop is needed
    if state.get("tool") == "human_in_the_loop":
        st.session_state.waiting_for_human_input = True
    else:
        # Add assistant message if available
        if isinstance(state.get("message"), list) and len(state["message"]) > 0:
            assistant_message = state["message"][-1]
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        
        # Check if there's an interrupt
        if "__interrupt__" in state:
            st.session_state.waiting_for_human_input = True

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow" not in st.session_state:
    st.session_state.workflow = create_workflow()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = 1

if "waiting_for_human_input" not in st.session_state:
    st.session_state.waiting_for_human_input = False

if "reasoning_steps" not in st.session_state:
    st.session_state.reasoning_steps = []

# App title and description
st.title("🧀 Cheese Agent")
st.markdown("Ask me anything about cheese products!")

with st.sidebar:
    st.title("Chat Options")
    if st.button("New Chat"):
        # Clear conversation history
        st.session_state.messages = []
        st.session_state.reasoning_steps = []
        st.session_state.waiting_for_human_input = False
        st.session_state.thread_id += 1  # Increment thread ID for new conversation
        st.rerun()
    
    image_path = os.path.join("./src/image", "Capture.PNG")
    if os.path.exists(image_path):
        st.image(image_path, caption="Cheese Bot Graph Visualization")
        
        # Add a button to view image in modal
        if st.button("Show Graph Visualization"):
            # Set flag to show modal in main container
            st.session_state.show_modal = True
    else:
        st.warning(f"Image not found. Please make sure the file exists at: {image_path}")

# Initialize modal state if not exists
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

# Display modal in main container if button was clicked
if st.session_state.show_modal:
    image_path = os.path.join("./src/image", "Capture.PNG")
    if os.path.exists(image_path):
        import base64
        
        # Read the image file
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            
        # Encode the image as base64
        b64_img = base64.b64encode(img_data).decode()
        
        # Create columns for the close button positioning
        close_col1, close_col2 = st.columns([1, 0.05])
        
        with close_col2:
            # Create a close button at the top right with an X icon
            if st.button("❌", key="close_modal_btn"):
                st.session_state.show_modal = False
                st.rerun()
                
        # Create HTML for a modal-like display in main container
        modal_html = f"""
        <style>
            .modal-container {{
                display: block;
                position: relative;
                width: 100%;
                height: 80vh;
                background-color: rgba(243, 239, 239, 0.9);
                overflow: auto;
                border-radius: 10px;
                text-align: center;
                margin-top: 10px;
                
            }}
            .modal-content {{
                margin: auto;
                display: block;
                border-radius: 5px;
                max-height: 90%;
                margin-top: 20px;
            }}
        </style>
        <div class="modal-container">
            <img class="modal-content" src="data:image/png;base64,{b64_img}">
        </div>
        """
        # Display modal in main container
        st.components.v1.html(modal_html, height=800)

# Display chat messages in conversation order
for message in st.session_state.messages:
    print(st.session_state.messages)
    with st.chat_message(message["role"]):
        if message["role"] == "human":
            st.write(message['content'])
        else:
            st.write(message['content'].content)
        
# Display reasoning steps in an expandable section
if st.session_state.reasoning_steps:
    with st.expander("Reasoning Steps", expanded=False):
        for step in st.session_state.reasoning_steps:
            st.markdown(f"**{step['stage']}**: {step['content']}")

# Human-in-the-loop section
if st.session_state.waiting_for_human_input:
    with st.container():
        st.warning("The bot needs more information to proceed.")
        human_feedback = st.text_input("Your feedback:", key="human_feedback_input")
        if st.button("Submit Feedback"):
            if human_feedback:
                # Add human feedback to chat
                st.session_state.messages.append({"role": "human", "content": f"Additional info: {human_feedback}"})
                
                # Add spinner to show thinking process
                with st.spinner("Processing your feedback..."):
                    # Resume workflow with human feedback
                    state = st.session_state.workflow.invoke(
                        Command(resume=[{"args": human_feedback}]), 
                        config={"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    
                    # Process the response
                    process_workflow_response(state)
                
                # Reset waiting state
                st.session_state.waiting_for_human_input = False
                
                # Rerun to update UI
                st.rerun()



# Chat input
if not st.session_state.show_modal:
    if not st.session_state.waiting_for_human_input:
        user_query = st.chat_input("Ask about cheese products...")
        
        if user_query:
            # Add user message to chat
            st.session_state.messages.append({"role": "human", "content": user_query})
            
            # Clear previous reasoning steps for new query
            st.session_state.reasoning_steps = []
                
            with st.chat_message(st.session_state.messages[-1]["role"]):
                if st.session_state.messages[-1]["role"] == "human":
                    st.write(st.session_state.messages[-1]['content'])

            
            # Run workflow with user query
            initial_state = {
                "message": [user_query],
                "aggregated_context": "",
                "curr_context": "",
                "query_to_retrieve_or_answer": "",
                "tool": "",
                "curr_state": "",
                "answer_quality": "GOOD",
                "human_feedback": ""
            }
            
            with st.spinner("Thinking..."):
                state = st.session_state.workflow.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )
                
                # Process the response
                process_workflow_response(state)
                
            # Rerun to update UI
            st.rerun()
