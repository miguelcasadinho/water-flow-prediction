import os
import logging
import schedule
import time
import subprocess

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up logging to file and console
log_file_path = "/home/giggo/python/logs/za1.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),    # Log to a file
        logging.StreamHandler()                # Also log to the console
    ]
)

# Function to run a specified Python script and log errors
def run_script(script_path):
    try:
        logging.info(f"Starting the process for {script_path}...")
        # Create the command to activate the virtual environment and run the script
        command = f'/home/giggo/python/Neural-Networks/nn-env/bin/python3 {script_path}'
        # Use shell=True to run the command in the shell
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        
        if result.returncode == 0:
            logging.info(f"Script {script_path} executed successfully.")
        else:
            logging.error(f"Script {script_path} failed with error:\n{result.stderr}")
    except FileNotFoundError:
        logging.error(f"Script not found: {script_path}")
    except Exception as e:
        logging.error(f"An error occurred while executing {script_path}: {e}")

# Schedule the dynamic model update process
schedule.every().day.at("01:00").do(lambda: run_script('/home/giggo/python/water-flow-prediction/za/zmc_za1/rnn_update.py'))
schedule.every().day.at("01:01").do(lambda: run_script('/home/giggo/python/water-flow-prediction/za/zmc_za1/rnn_predict.py'))

# Infinite loop to keep the script running and checking for scheduled jobs
if __name__ == "__main__":
    logging.info("Scheduler started. Waiting for scheduled tasks...")
    try:
        while True:
            schedule.run_pending()  # Check if any scheduled tasks are pending to run
            time.sleep(1)  # Sleep for a short while before checking again
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user.")