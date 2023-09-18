import numpy as np
import numba


@numba.njit()
def get_sum(bi_tree, i):
    s = 0 
    # index in BITree[] is 1 more than the index in arr[]
    i = i+1
  
    # Traverse ancestors of BITree[index]
    while i > 0:
        # Add current element of BITree to sum
        s += bi_tree[i]  
        # Move index to parent node in getSum View
        i -= i & (-i)
    return s

@numba.njit()
def get_range_sum(bi_tree, left, right):
    # right inclusive
    return get_sum( bi_tree, right) - get_sum(bi_tree, left - 1)
  
@numba.njit()
def updatebit(bi_tree, n ,i, v):
    """
    updates the bi_tree, is equivalent to adding v to the
    value at position i in the original array.
    """
    i += 1
  
    # Traverse all ancestors and add 'val'
    while i <= n:  
        # Add 'val' to current node of BI Tree
        bi_tree[i] += v
        # Update index to that of parent in update View
        i += i & (-i)
    
    return 0

@numba.njit()
def construct(arr):
    n = arr.size
    # Create and initialize BITree[] as 0
    bi_tree = np.zeros(n+1, dtype=np.float64)
    # Store the actual values in BITree[] using update()
    for i in range(n):
        updatebit(bi_tree, n, i, arr[i])
    # Uncomment below lines to see contents of BITree[]
    #for i in range(1,n+1):
    #     print BITTree[i],
    return bi_tree