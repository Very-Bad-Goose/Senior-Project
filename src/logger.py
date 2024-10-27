import time
from datetime import datetime
import google_sheet
import os

"""
class SheetLogger:
    def __init__(self, google_sheet_instance, log_sheet_name="Log"):
        self.google_sheet_instance = google_sheet_instance
        self.log_sheet = self.google_sheet_instance.check_or_create_log_sheet(log_sheet_name)
    
    def log(self, operation_name, result, details=None):
        #Logs information about an operation to the log sheet.
        try:
            if not isinstance(result, dict):
                raise ValueError(f"Expected result to be dictionary but got {type(result)}")
            # Capture the current time and date
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate execution time if result contains it
            exec_time = result.get('execution_time', 'N/A')
            
            # Format the log message
            log_message = [
                timestamp,                         # Time of log entry
                operation_name,                    # Operation being called
                exec_time,                         # Execution time
                result.get('status', 'Unknown'),   # Status of the operation
                result.get('message', ''),         # Any result message or detail
                details                            # Any additional details or metadata (optional)
            ]
            
            # Log the message content for debugging
            print(f"Logging data: {log_message}")
            
            # Append the log message to the log sheet
            self.google_sheet_instance.log_result(log_message)
            print(f"Logged: {log_message}")
        
        except Exception as e:
            print(f"Error during logging: {e}")
    
    def log_failure(self, operation_name, error_message):
        # Handles logging for failures
        exec_time = "N/A"   # No execution time if there's a failure
        self.log(operation_name, {
            'status': 'Failure',
            'message': error_message,
            'execution_time': exec_time
        })
        
    def time_operation(self, func, *args, **kwargs):
        # Decorator-like method to log the execution time of an operation.
        start_time = time.time()
        try:
            # Execute the function and capture the result
            result = func(*args, **kwargs)
            
            # Calculate the total execution time
            exec_time = round(time.time() - start_time, 4)
            return {
                'status': 'Success',
                'message': result,
                'execution_time': exec_time
            }
        except Exception as e:
            exec_time = round(time.time() - start_time, 4)
            return {
                'status': 'Failure',
                'message': str(e),
                'execution_time': exec_time
            }
            
            """
class FileLogger:
    def __init__(self, log_dir="logs"):
        """
        Initialize the logger. Creates a daily log file in the specified directory.
        Args:
            log_dir (str): Directory where logs will be stored.
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file_path = self.get_log_file_path()

    def get_log_file_path(self):
        """Generate a file path for the log file based on the current date."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"log_{current_date}.txt")

    def log(self, operation_name, result, details=None):
        """
        Logs information about an operation to the daily log file.
        Args:
            operation_name (str): Name of the operation.
            result (dict): Result information, must include status and may include execution time and message.
            details (str, optional): Additional details about the operation.
        """
        try:
            if not isinstance(result, dict):
                raise ValueError(f"Expected result to be dictionary but got {type(result)}")
                
            # Capture the current timestamp for the log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            exec_time = result.get('execution_time', 'N/A')
            log_entry = (
                f"{timestamp} | Operation: {operation_name} | Execution Time: {exec_time}s | "
                f"Status: {result.get('status', 'Unknown')} | Message: {result.get('message', '')}"
            )
            if details:
                log_entry += f" | Details: {details}"

            # Write the log entry to the daily log file
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(log_entry + "\n")
                
            print(f"Logged: {log_entry}")

        except Exception as e:
            print(f"Error during logging: {e}")

    def log_failure(self, operation_name, error_message):
        """Logs a failure event with the error message."""
        self.log(operation_name, {
            'status': 'Failure',
            'message': error_message,
            'execution_time': 'N/A'
        })

    def time_operation(self, func, *args, **kwargs):
        """
        Times the execution of an operation and logs its performance.
        Args:
            func (callable): The function to be executed and timed.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        Returns:
            dict: Result of the function execution with status, message, and execution time.
        """
        start_time = time.time()
        try:
            # Execute the function and capture the result
            result = func(*args, **kwargs)
            exec_time = round(time.time() - start_time, 4)
            return {
                'status': 'Success',
                'message': result,
                'execution_time': exec_time
            }
        except Exception as e:
            exec_time = round(time.time() - start_time, 4)
            return {
                'status': 'Failure',
                'message': str(e),
                'execution_time': exec_time
            }
            