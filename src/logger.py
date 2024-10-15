import time
from datetime import datetime
import google_sheet

class SheetLogger:
    def __init__(self, google_sheet_instance, log_sheet_name="Log"):
        self.google_sheet_instance = google_sheet_instance
        self.log_sheet = self.google_sheet_instance.check_or_create_log_sheet(log_sheet_name)
    
    def log(self, operation_name, result, details=None):
        """Logs information about an operation to the log sheet."""
        try:
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
            
            # Append the log message to the log sheet
            self.google_sheet_instance.log_result(log_message)
            print(f"Logged: {log_message}")
        
        except Exception as e:
            print(f"Error during logging: {e}")
    
    def log_failure(self, operation_name, error_message):
        """Handles logging for failures"""
        exec_time = "N/A"   # No execution time if there's a failure
        self.log(operation_name, {
            'status': 'Failure',
            'message': error_message,
            'execution_time': exec_time
        })
        
    def time_operation(self, func, *args, **kwargs):
        """Decorator-like method to log the execution time of an operation."""
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