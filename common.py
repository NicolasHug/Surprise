lastMi = 1682
lastUi = 943

class Col:
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

#1− | (a − b) − (c − d) | if a ≥ b and c ≥ d, or a ≤ b and c ≤ d
#1 − max(| a − b |,| c − d |) if a ≤ b and c ≥ d, or a ≥ b and c ≤ d
def tvA(a, b, c, d):
    if (a >= b and c >= d) or (a <=b and c <= d):
        return 1 - abs((a-b) - (c-d))
    else:
        return 1 - max(abs(a-b), abs(c-d))


def solveAstar(a, b, c):
    """ solve A*(a, b, c, x). Undefined if equation not solvable."""
    return c if a == b else b

def idty(a, b, c, d):
    return 1 - (max(a, b, c, d) - min(a, b, c, d))

def diff(x, y):
    return abs(x - y)

def eq(x, y, z):  
    return 1 - abs(max(x, y, z) - min(x, y, z))

def hd1(a, b, c, d):
    return max(min(eq(a, b, d), diff(d, c)), min(eq(a, c, d), diff(d, b)),
            min(eq(b, c, d), diff(d, a)))    

