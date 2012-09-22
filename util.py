import scipy.linalg as linalg
import scipy
import numpy as np

def is_singular(A, eps=1e-15, verbose=False):
    """This method returns whether or not a matrix is singular to a precision given by eps

    @param A: The matrix to check for singularity
    @type A: array
    @param eps: The precision to which to determine if A is singular
    @type eps: float
    @return: Whether or not A is singular
    @rtype: boolean
    """
    u, s, vh = linalg.svd(A)
    #singular values are in s. get the maximum and minimum
    max_s = np.max(np.abs(s))
    min_s = np.min(np.abs(s))
    ratio = min_s / max_s
    sing = ratio < eps
    if verbose:
        print "Checking singularity: matrix has max s'value %s, min s'value %s, ratio = %s. Singularity = %s" % (
            max_s, min_s, ratio, sing)
    return sing

def null_vector(A):
    """Gets the nullspace of a matrix A, assuming that it is singular and has exactly one singular value equal to zero.
    Preface this around a call to is_singular

    @param A: The maxtix whose nullspace to get
    @type A: array
    @return A vector representing A's nullspace
    @rtype: array
    """
    u, s, vh = linalg.svd(A)

    vht = vh.T.conj()
    #find min s
    col = np.where(s == np.min(s))[0][0]
    return vht[:,col]

def null(A, eps=1e-15):
    """Returns the null-space of the matrix A
    Implementation from
    http://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    """
    u, s, vh = linalg.svd(A)
    null_mask = (s < eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)