import datetime


def now() -> str:
    """Return current time as YYYY-MM-DD_HH-MM-SS

    Returns
    -------
    string
        current time
    """

    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
