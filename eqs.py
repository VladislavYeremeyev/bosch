"""
    File contains functions define equvivalence relation on the set of structures
"""
def idEq(a, b):
    """
        The most stringent equvivalence relation

        Parameters:
        --------------
        a : type?
            Structure type for checked for equvivalence
        b : type?
            Structure type for checked for equvivalence

        Return:
        --------------
        res : bool
            States equvivalence "a" and "b" or not
    """
    return a == b

def trueEq(a, b):
    """
        The least stringent equvivalence relation

        Parameters:
        --------------
        a : type?
            Structure type for checked for equvivalence
        b : type?
            Structure type for checked for equvivalence

        Return:
        --------------
        res : bool
            States equvivalence "a" and "b" or not
    """
    return True

