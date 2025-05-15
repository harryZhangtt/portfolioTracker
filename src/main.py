import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now the imports will work
from src.models.retail_tracker import RetailTracker
from src.controller.command_line import CommandLine

if __name__ == "__main__":
    # Make sure the src and database directories exist
    src_dir = os.path.dirname(os.path.abspath(__file__))
    database_dir = os.path.join(os.path.dirname(src_dir), "database")
    os.makedirs(database_dir, exist_ok=True)
    
    # Create tracker instance and run initial update
    tracker = RetailTracker()
    tracker.run_tracker()
    
    # Start the command line interface
    cli = CommandLine(tracker)
    cli.run()
