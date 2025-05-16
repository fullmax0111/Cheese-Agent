import os
import subprocess
import sys

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print("Starting Cheese Bot Streamlit App...")
    
    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", "src/app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down Cheese Bot...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")      
        sys.exit(1) 