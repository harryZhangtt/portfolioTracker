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
        raise TimeoutError

    def run(self):
        print("Command Line Interface. Available commands: transaction, rerender, exit, import, export, addFund, extractFund, analyze")
        signal.signal(signal.SIGALRM, self.timeout_handler)
        
        while True:
            try:
                # Set alarm for 30 seconds
                signal.alarm(30)
                print("\nCommand> ", end='', flush=True)
                cmd = input().strip()
                
                # Immediately reset the alarm to avoid timeout during command execution
                signal.alarm(0)
                
                if cmd == "exit":
                    print("Exiting command line.")
                    break
                elif cmd in self.commands:
                    try:
                        self.commands[cmd].execute()
                    except Exception as e:
                        print(f"Error executing command {cmd}: {e}")
                    finally:
                        # Make sure matplotlib doesn't keep any windows open
                        plt.close('all')
                        
                    print("\nCommand completed. Ready for next command.")
                else:
                    print("Unknown command. Please try again.")
                
                # Reset the alarm after command execution is complete
                signal.alarm(30)
            except TimeoutError:
                print("\nNo command received in 30 seconds. Terminating command line.")
                break
            except KeyboardInterrupt:
                print("\nCommand interrupted. Returning to command prompt.")
                signal.alarm(30)  # Reset the alarm after an interruption
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                signal.alarm(30)  # Reset the alarm after any error
