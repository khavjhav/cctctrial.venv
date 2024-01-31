import multiprocessing

# Define the worker function for each camera
def run_worker(channel_id):
    # Import necessary modules here
    # Your worker code here (similar to the code in your worker.py)

    if __name__ == "__main__":
    # List of channel IDs for all cameras
      channel_ids = ["1902", "3202" ]  # Add more channel IDs as needed

    processes = []
    
    # Create and start a process for each camera
    for channel_id in channel_ids:
        process = multiprocessing.Process(target=run_worker, args=(channel_id,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
