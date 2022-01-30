import inspect


def indented_print(*args, **kwargs):
    print("  " * len(inspect.stack()), end="")
    return print(*args, **kwargs)
