""" https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
import socket
import sys
from contextlib import closing
with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    FOUND_FREE_PORT = True
    sys.exit(str(s.getsockname()[1]))