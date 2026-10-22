"""Bound native simulator teardown so a finished/failed client cannot hang forever."""
from contextlib import contextmanager
import os
import sys
import threading
import traceback


@contextmanager
def bounded_cleanup(timeout=60.0, exit_code=1):
    """Force process exit if native cleanup does not return before the deadline."""
    if timeout <= 0:
        raise ValueError('cleanup timeout must be positive')
    def expired():
        # Avoid buffered Python streams: their lock may be held by stuck teardown.
        try:
            os.write(2, b'[eval] native cleanup timed out; forcing client exit\n')
        finally:
            os._exit(exit_code)
    timer = threading.Timer(timeout, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def run_evaluation(main, abort_videos, *, cleanup_timeout=60.0):
    """Keep the traceback and exit code, but bypass Kit's failing atexit hooks.

    A supervisor can retry using the existing run ID/resume manifest. Never
    turn an infrastructure exception into an evaluated or abandoned layout.
    """
    try:
        main()
    except BaseException as error:
        if isinstance(error, SystemExit):
            code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        elif isinstance(error, KeyboardInterrupt):
            code = 130
        else:
            code = 1
        # Include diagnostics and stream flushing in the deadline as well.
        with bounded_cleanup(cleanup_timeout, exit_code=code):
            if not isinstance(error, SystemExit):
                traceback.print_exc()
            try:
                abort_videos()
            except Exception:
                traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
