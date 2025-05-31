import signal
import matplotlib.pyplot as plt
# Use absolute imports instead of relative imports
from src.commands.command_base import Command
from src.commands.transaction_commands import TransactionCommand
from src.commands.portfolio_commands import ImportPortfolioCommand, ExportFileCommand, RerenderCommand
from src.commands.fund_commands import AddFundCommand, ExtractFundCommand
from src.commands.analysis_commands import ExcessReturnCommand
from src.commands.utility_commands import UtilCommand

class CommandLine:
    def __init__(self, tracker):
        self.tracker = tracker
        self.commands = {
            "transaction": TransactionCommand(tracker),
            "rerender": RerenderCommand(tracker),
            "import": ImportPortfolioCommand(tracker),
            "export": ExportFileCommand(tracker),
            "addFund": AddFundCommand(tracker),
            "extractFund": ExtractFundCommand(tracker),
            "util": UtilCommand(tracker),
            "analyze": ExcessReturnCommand(tracker)
        }

    def timeout_handler(self, signum, frame):
        """Handle timeout by raising TimeoutError"""
        print("\nNo command received in 30 seconds. Terminating command line.")
        raise TimeoutError("Command input timed out")

    def run(self):
        print("Command Line Interface. Available commands: transaction, rerender, exit, import, export, addFund, extractFund, analyze")
        
        # Set up timeout handler
        signal.signal(signal.SIGALRM, self.timeout_handler)
        
        while True:
            try:
                # Set alarm for 30 seconds before each input prompt
                signal.alarm(5)  # Fixed timeout to 30 seconds (was 3)
                
                print("\nCommand> ", end='', flush=True)
                cmd = input().strip()
                
                # Cancel alarm immediately after receiving input
                signal.alarm(0)
                
                if cmd == "exit":
                    print("Exiting command line.")
                    break
                elif cmd in self.commands:
                    try:
                        # Execute the command (with no active alarm)
                        self.commands[cmd].execute()
                    except Exception as e:
                        print(f"Error executing command {cmd}: {e}")
                    finally:
                        # Make sure matplotlib doesn't keep any windows open
                        plt.close('all')
                        # Ensure signal handler is set back to our timeout_handler
                        signal.signal(signal.SIGALRM, self.timeout_handler)
                        
                    print("\nCommand completed. Ready for next command.")
                else:
                    print("Unknown command. Please try again.")
                
                # We intentionally don't set an alarm here - it gets set at the beginning of the loop
                
            except TimeoutError:
                # Timeout is handled by the timeout_handler function
                break
            except KeyboardInterrupt:
                print("\nCommand interrupted. Returning to command prompt.")
            except Exception as e:
                print(f"\nUnexpected error: {e}")
