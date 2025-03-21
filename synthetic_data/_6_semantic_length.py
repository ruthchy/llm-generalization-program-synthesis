import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

class ExecutionTimeLength:
    """
    Measures the execution time of Logo programs as a length metric.
    This provides a runtime-based complexity measure that complements semantic length.
    """

    def __init__(self, timeout=10, num_runs=3):
        """
        Initialize the ExecutionTimeLength calculator.
        
        Args:
            timeout (int): Maximum execution time in seconds before timing out
            num_runs (int): Number of times to execute program for averaging
        """
        self.timeout = timeout
        self.num_runs = num_runs
        
        # Get repository root directory
        current_dir = os.getcwd()

        # Add dependencies path to sys.path if not already there
        dependencies_path = os.path.join(current_dir, 'external/dependencies')
        if dependencies_path not in sys.path:
            sys.path.append(dependencies_path)

    @contextmanager
    def time_limit(self, seconds):
        """
        Context manager to limit execution time.
        
        Args:
            seconds (int): Maximum execution time in seconds
        """
        import signal
        
        def signal_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {seconds} seconds")
            
        # Set the timeout handler
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)

    def calc_execution_time(self, program):
        """
        Calculate execution time of a Logo program.
        
        Args:
            program (str): Logo program code
            
        Returns:
            float: Average execution time in seconds or float('inf') if execution fails
        """
        # Import necessary modules for turtle execution
        try:
            from program_refactoring.domains.logos.pyturtle_pc import PyTurtle, HALF_INF, EPS_DIST, EPS_ANGLE
        except ImportError:
            print("Error: Could not import PyTurtle modules. Make sure the dependencies are correctly set up.")
            return float('inf')
        
        execution_times = []
        
        for run in range(self.num_runs):
            # Create PyTurtle instance
            turtle = PyTurtle()  # Disable drawing to speed up execution
            
            # Create execution scope with all necessary variables
            exec_scope = {
                "turtle": turtle,
                "HALF_INF": HALF_INF,
                "EPS_DIST": EPS_DIST,
                "EPS_ANGLE": EPS_ANGLE,
                "forward": turtle.forward,
                "left": turtle.left,
                "right": turtle.right,
                "teleport": turtle.teleport,
                "penup": turtle.penup,
                "pendown": turtle.pendown,
                "heading": turtle.heading,
                "embed": turtle.embed
            }
            
            try:
                with self.time_limit(self.timeout):
                    # Time the execution
                    start_time = time.time()
                    exec(program, exec_scope)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
            except TimeoutError:
                print(f"Program execution timed out after {self.timeout} seconds.")
                return float('inf')
            except Exception as e:
                print(f"Error executing program: {str(e)}")
                return float('inf')
            finally:
                # Clean up
                plt.close(turtle.fig)  # Close the figure to free up memory
                if 'turtle' in locals():
                    del turtle


        # Return the average execution time if we have valid measurements
        if execution_times:
            # Remove outliers using interquartile range
            if len(execution_times) >= 3:
                q1 = np.percentile(execution_times, 25)
                q3 = np.percentile(execution_times, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_times = [t for t in execution_times if lower_bound <= t <= upper_bound]
                if filtered_times:
                    return np.mean(filtered_times)
            
            # If we can't filter or only have a few measurements, return simple average
            return np.mean(execution_times)
        else:
            return float('inf')