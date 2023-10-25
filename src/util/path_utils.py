import os

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.pardir))

def project_path(*args) -> str:
    """
    Abstraction from os.path.join()
    Builds absolute paths from relative path strings with package as root.
    If args already contains an absolute path, it is used as root for the subsequent joins
    Args:
        *args:

    Returns:
        absolute path

    """

    return os.path.abspath(os.path.join(PACKAGE_DIR, *args))
