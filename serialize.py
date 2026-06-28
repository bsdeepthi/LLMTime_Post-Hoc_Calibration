from dataclasses import dataclass, field
import numpy as np


@dataclass
class SerializerSettings:
    base: int = 10
    prec: int = 3
    signed: bool = True
    time_sep: str = ', '
    bit_sep: str = ''
    minus_sign: str = '-'
    plus_sign: str = ''
    decimal_point: str = '.'
    missing_str: str = 'NaN'


def _to_base_digits(n: int, base: int, width: int) -> list[int]:
    """Convert non-negative integer to fixed-width digit list in given base."""
    digits = []
    for _ in range(width):
        digits.append(n % base)
        n //= base
    return digits[::-1]


def serialize_num(x: float, settings: SerializerSettings) -> str:
    if np.isnan(x):
        return settings.missing_str

    if settings.signed and x < 0:
        sign = settings.minus_sign
        x = -x
    else:
        sign = settings.plus_sign

    base, prec = settings.base, settings.prec

    if base == 10:
        # Standard decimal: format to prec decimal places
        s = f"{x:.{prec}f}"
        if settings.bit_sep:
            int_part, frac_part = s.split('.')
            s = int_part + settings.decimal_point + settings.bit_sep.join(frac_part)
        elif settings.decimal_point != '.':
            s = s.replace('.', settings.decimal_point)
    else:
        # Arbitrary base: represent scaled integer, then split into int/frac digits
        scale = base ** prec
        scaled = int(round(x * scale))
        int_val = scaled // scale
        frac_val = scaled % scale

        int_digits = []
        tmp = int_val
        if tmp == 0:
            int_digits = [0]
        while tmp > 0:
            int_digits.append(tmp % base)
            tmp //= base
        int_digits = int_digits[::-1]

        frac_digits = _to_base_digits(frac_val, base, prec)

        sep = settings.bit_sep
        int_str = sep.join(str(d) for d in int_digits)
        frac_str = sep.join(str(d) for d in frac_digits)
        s = int_str + settings.decimal_point + frac_str

    return sign + s


def serialize_arr(arr, settings: SerializerSettings) -> str:
    return settings.time_sep.join(serialize_num(x, settings) for x in arr)


def deserialize_num(s: str, settings: SerializerSettings) -> float:
    s = s.strip()
    if s == settings.missing_str:
        return np.nan

    if settings.minus_sign and s.startswith(settings.minus_sign):
        sign = -1
        s = s[len(settings.minus_sign):]
    else:
        sign = 1
        if settings.plus_sign and s.startswith(settings.plus_sign):
            s = s[len(settings.plus_sign):]

    if settings.bit_sep:
        s = s.replace(settings.bit_sep, '')

    if settings.decimal_point != '.':
        s = s.replace(settings.decimal_point, '.')

    if settings.base == 10:
        return sign * float(s)

    # Arbitrary base
    if '.' in s:
        int_part, frac_part = s.split('.')
    else:
        int_part, frac_part = s, ''

    int_val = 0
    for ch in int_part:
        int_val = int_val * settings.base + int(ch, settings.base)

    frac_val = 0.0
    for i, ch in enumerate(frac_part):
        frac_val += int(ch, settings.base) / (settings.base ** (i + 1))

    return sign * (int_val + frac_val)


def deserialize_str(s: str, settings: SerializerSettings, ignore_last: bool = False, steps: int = None):
    """
    Parse an LLM completion string back to a numpy array.
    Stops at the first token that cannot be parsed.
    Returns None if no values could be parsed.
    """
    parts = s.split(settings.time_sep)

    if ignore_last and parts:
        parts = parts[:-1]

    values = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            values.append(deserialize_num(part, settings))
        except (ValueError, TypeError):
            break  # LLM may append explanation text — stop here

    if not values:
        return None

    arr = np.array(values)
    if steps is not None:
        arr = arr[:steps]

    return arr if len(arr) > 0 else None
