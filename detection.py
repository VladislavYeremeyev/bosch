"""
    File contains function for detection structure
"""

from eqs import idEq, trueEq

class StructType:
    def __init__(self, sType, sCategory):
        """
            Define struct  type


            Parameters:
            -----------------
            sType : type?
                Type of structure
            sCategory : type?
                Category of structure
        """
        self.sType = sType
        self.sCategory = sCategory

class Structure:

    def __init__(self, obj, coord, structType):
        """
            Define structure

            Parameters:
            -----------------
            obj : type?
                Visual representation of structure
            coord : type?
                Structure coordinate on the image
            structType : StructType
                Type of structure
        """

        self.obj        = obj
        self.coord      = coord
        self.structType = structType


def detectStructures(imagePath):
    """
        Get all structures from an image

        Parameters:
        --------------------
        imagePath : type?
            Path to the folder with images

        Return:
        --------------------
        structures : collection of structures
            Set of structure from the images
    """

    return None

def detectRealityStructures(folderPath):
    """
        Get all structures from all the images from the given folder

        Parameters:
        --------------------
        folderPath : type?
            Path to the folder with images

        Return:
        --------------------
        structures : collection of structures
            Set of structure from the images
    """

    return None

def isEqual(eqFun, a, b):
    """
        Check equals of structure depend on given equality relation  

        Parameters:
        ------------------
        eqFun : function(a,b) -> bool
            Equality relation
        a : type?
            Structure for checked for equvivalence
        b : type?
            Structure for checked for equvivalence

        Return:
        --------------
        res : bool
            States equvivalence types "a" and "b" or not
    """

    return eqFun(a.structType, b.structType)

