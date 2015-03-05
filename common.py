lastMi = 1682 # last movie index for u1.base
lastUi = 943 # last user index for u1.base

class Col:
    """A class for adding color in the term output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def tvAStar (a, b, c, d):
    """return the truth value of A*(a, b, c, d)"""
    return min(1 - abs(max(a, d) - max(b, c)), 1 - abs(min(a, d) - min(b, c)))

def tvA(a, b, c, d):
    """return the truth value of A(a, b, c, d)"""
    if (a >= b and c >= d) or (a <=b and c <= d):
        return 1 - abs((a-b) - (c-d))
    else:
        return 1 - max(abs(a-b), abs(c-d))


def solveAstar(a, b, c):
    """ solve A*(a, b, c, x). Undefined if equation not solvable."""
    return c if a == b else b
