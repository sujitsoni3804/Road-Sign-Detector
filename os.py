import os

# Directories to exclude
excluded_dirs = {'Dataset', 'runs'}

# Open the log file
with open('file_log.txt', 'w') as log_file:
    for root, dirs, files in os.walk('.'):
        # Modify dirs in-place to exclude specified directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        # Log the current directory
        log_file.write(f"{root}/\n")
        
        # Log files in the current directory
        for file in files:
            log_file.write(f"{os.path.join(root, file)}\n")
