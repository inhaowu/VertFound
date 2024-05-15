import contextlib
import platform
import threading

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    # Join all daemon threads, i.e. atexit.register(lambda: join_threads())
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f'Joining thread {t.name}')
            t.join()