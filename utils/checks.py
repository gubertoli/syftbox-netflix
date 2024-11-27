import os
from datetime import datetime, date

def is_file_modified_today(file_path):
    """
    Checks if a file was created or modified today.

    :param file_path: Path to the file
    :return: True if the file was created or modified today, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"The file '{file_path}' does not exist.")
        return False

    # Get the file's last modification time
    file_mod_time = os.path.getmtime(file_path)
    
    # Convert the timestamp to a date
    file_mod_date = datetime.fromtimestamp(file_mod_time).date()

    # Compare with today's date
    return file_mod_date == date.today()


