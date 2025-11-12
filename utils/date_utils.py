from datetime import datetime

def get_current_date():
    """Return date in ISO format usable for prompts/searches."""
    return datetime.now().strftime("%Y-%m-%d")
