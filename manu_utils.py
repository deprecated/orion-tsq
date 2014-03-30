"""
Miscellaneous routines to support the PPAK spectra analysis
"""
TRANSLATION_TABLE = {
    ord(" "): '_',
    ord(","): None,
    ord("["): None,
    ord("]"): None
}

def sanitize_string(s):
    """Make string suitable to be a unix filename
Remove and/or substitute problematic characters via TRANSLATION_TABLE"""
    return s.translate(TRANSLATION_TABLE)

