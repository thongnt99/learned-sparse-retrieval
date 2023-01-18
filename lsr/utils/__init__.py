# Ref: https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python


import importlib


def get_absolute_class_name(o):
    "Return the full class path of an object"
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def get_class_from_str(class_str):
    "Instantiate a class objection from class name"
    chunks = class_str.split(".")
    module_str = ".".join(chunks[:-1])
    class_str = chunks[-1]
    module = importlib.import_module(module_str)
    return getattr(module, class_str)
