import time
from app_kp import app, scheduler, initialize_app_core

print("--- Starting Background Worker ---")

# The worker needs the application context to run jobs
with app.app_context():
    # Check if the scheduler is already running (it shouldn't be in this process)
    if not scheduler.running:
        print("Scheduler not running, starting now.")
        # We don't need to initialize everything, just start the scheduler.
        # The _schedule_jobs() function inside initialize_app_core will handle it.
        initialize_app_core() 
    else:
        print("Scheduler is already running.")

print("--- Worker is running, keeping process alive... ---")
# Keep the worker script alive indefinitely
try:
    while True:
        time.sleep(3600)  # Sleep for a long time, the scheduler runs in its own thread.
except (KeyboardInterrupt, SystemExit):
    # Shut down the scheduler when the worker is stopped
    scheduler.shutdown()
    print("--- Worker has been shut down. ---")