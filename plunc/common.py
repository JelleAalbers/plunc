import numpy as np

# TODO: instead of rounding to digits, make a uniform log space where we snap to
def round_to_digits(x, n_digits):
    """Rounds x to leading digits"""
    x = float(x)   # Doesn't work on numpy floats
    return round(x, n_digits - 1 - int(np.log10(x)) + (1 if x < 1 else 0))