"""
A set of utility functions for general python usage
"""
import inspect
from copy import deepcopy

import numpy as np


def merge_nested_dicts(base_dict, extra_dict, verbose=False):
    """
    Iteratively updates @base_dict with values from @extra_dict. Note: This generates a new dictionary!
    Args:
        base_dict (dict): Nested base dictionary, which should be updated with all values from @extra_dict
        extra_dict (dict): Nested extra dictionary, whose values will overwrite corresponding ones in @base_dict
        verbose (bool): If True, will print when keys are mismatched
    Returns:
        dict: Updated dictionary
    """
    # Loop through all keys in @extra_dict and update the corresponding values in @base_dict
    base_dict = deepcopy(base_dict)
    for k, v in extra_dict.items():
        if k not in base_dict:
            base_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(base_dict[k], dict):
                base_dict[k] = merge_nested_dicts(base_dict[k], v)
            else:
                not_equal = base_dict[k] != v
                if isinstance(not_equal, np.ndarray):
                    not_equal = not_equal.any()
                if not_equal and verbose:
                    print(f"Different values for key {k}: {base_dict[k]}, {v}\n")
                base_dict[k] = np.array(v) if isinstance(v, list) else v

    # Return new dict
    return base_dict


def get_class_init_kwargs(cls):
    """
    Helper function to return a list of all valid keyword arguments (excluding "self") for the given @cls class.
    Args:
        cls (object): Class from which to grab __init__ kwargs
    Returns:
        list: All keyword arguments (excluding "self") specified by @cls __init__ constructor method
    """
    return list(inspect.signature(cls.__init__).parameters.keys())[1:]


def extract_subset_dict(dic, keys, copy=False):
    """
    Helper function to extract a subset of dictionary key-values from a current dictionary. Optionally (deep)copies
    the values extracted from the original @dic if @copy is True.
    Args:
        dic (dict): Dictionary containing multiple key-values
        keys (Iterable): Specific keys to extract from @dic. If the key doesn't exist in @dic, then the key is skipped
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys
    Returns:
        dict: Extracted subset dictionary containing only the specified @keys and their corresponding values
    """
    subset = {k: dic[k] for k in keys if k in dic}
    return deepcopy(subset) if copy else subset


def extract_class_init_kwargs_from_dict(cls, dic, copy=False):
    """
    Helper function to return a dictionary of key-values that specifically correspond to @cls class's __init__
    constructor method, from @dic which may or may not contain additional, irrelevant kwargs.
    Note that @dic may possibly be missing certain kwargs as specified by cls.__init__. No error will be raised.
    Args:
        cls (object): Class from which to grab __init__ kwargs that will be be used as filtering keys for @dic
        dic (dict): Dictionary containing multiple key-values
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys
    Returns:
        dict: Extracted subset dictionary possibly containing only the specified keys from cls.__init__ and their
            corresponding values
    """
    # extract only relevant kwargs for this specific backbone
    return extract_subset_dict(
        dic=dic,
        keys=get_class_init_kwargs(cls),
        copy=copy,
    )


def assert_valid_key(key, valid_keys, name=None):
    """
    Helper function that asserts that @key is in dictionary @valid_keys keys. If not, it will raise an error.

    :param key: Any, key to check for in dictionary @dic's keys
    :param valid_keys: Iterable, contains keys should be checked with @key
    :param name: str or None, if specified, is the name associated with the key that will be printed out if the
        key is not found. If None, default is "value"
    """
    if name is None:
        name = "value"
    assert key in valid_keys, "Invalid {} received! Valid options are: {}, got: {}".format(name, valid_keys, key)
