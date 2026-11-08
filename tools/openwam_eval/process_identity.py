import subprocess,time
from pathlib import Path

class ExistingProcess:

    def __init__(self, cookie):
        self.cookie = cookie
        self.pid = int(cookie['pid'])
        if self.poll() is not None:
            raise RuntimeError(f'Adopted process {self.pid} is no longer live')

    def poll(self):
        p = Path(f'/proc/{self.pid}')
        try:
            fields = (p / 'stat').read_text().split(') ', 1)[1].split()
            if fields[19] != self.cookie['start']:
                return 1
            if (p / 'cgroup').read_text() != self.cookie['cgroup']:
                return 1
            return 0 if fields[0] in ('Z', 'X') else None
        except FileNotFoundError:
            return 0

    def wait(self, timeout=None):
        start = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - start > timeout:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.2)
        return self.poll()
