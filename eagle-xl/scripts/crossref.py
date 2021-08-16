"""Routines for matching indices between two data sets."""

import numpy as np
import os
import time
from velociraptor import load
import h5py as h5


class ReverseList:
    """Class for creating and querying a reverse-index list.

    This is essentially a thin wrapper around the `create_reverse_list()`
    function, but avoids the need to artificially expand the list
    to deal with possible out-of-bounds queries. It creates and queries
    an array containing the index for each ID value, i.e. the inverse of
    the input list (which gives the ID for each index).

    Warning
    -------
    It is a bad idea to instantiate this class with a key set that contains
    large values in relation to the available memory: as a rough guide,
    consider the limit as 1/8 * [RAM/byte]. Ignoring this will result in
    undefined, slow, and likely annoying behaviour.

    Parameters
    ----------
    ids : ndarray (int)
        Input keys (IDs) to invert. These are assumed to be unique;
        death and destruction may happen if this is not the case.
        The array must be one-dimensional. Any negative values are
        assumed to be dummy elements and are ignored.
    delete_ids : bool, optional
        Delete input keys after building reverse list (default: ``False``).
    assume_positive : bool, optional
        Assume that all the input values are non-negative, which speeds
        up the inversion. If ``False`` (default), the code checks
        explicitly which input values are non-negative and only
        transcribes these to the reverse list.
    compact : bool, optional
        Make the reverse-index list shorter by (transparently) subtracting
        the minimum ID from inputs (default: ``False``).

    Attributes
    ----------
    reverse_IDs : ndarray(int)
        The reverse-index array, created upon instantiation.
    num_int : int
        The number of keys in the input array.
    """

    def __init__(self, ids, delete_ids=False, assume_positive=False,
                 compact=False):
        self.num_int = len(ids)

        # Find and subtract minimum (positive) value from input keys
        if compact:
            if assume_positive:
                self.min_key = np.min(ids)
            else:
                ind_positive = np.nonzero(ids >= 0)[0]
                self.min_key = np.min(ids[ind_positive])
            self.reverse_IDs = create_reverse_list(
                ids - self.min_key, delete_ids=delete_ids,
                assume_positive=assume_positive)
        else:
            # Invert unmodified keys
            self.min_key = None
            self.reverse_IDs = create_reverse_list(
                ids, delete_ids=delete_ids, assume_positive=assume_positive)

    def query(self, ids, assume_valid=False):
        """Find the indices of the input (external) IDs.

        Parameters
        ----------
        ids : ndarray(int)
            The IDs whose indices in the internal list (used to set up the
            reverse list) should be determined. Out-of-bound situations are
            dealt with internally.
        assume_valid : bool, optional
            Assume that all input IDs are within the range of the internal
            reverse-index list, so that out-of-bound check can be skipped
            (default: ``False``).

        Returns
        -------
        indices : ndarray(int)
            The indices corresponding to each input ID (-1 if not found).

        Note
        ----
        This is also the object's call method, so it can be used
        directly without specifying `query`.

        Example
        -------
        >>> import numpy as np
        >>> ids = np.array([4, 0, 5, 1])
        >>> reverse_ids = ReverseList(ids)
        >>> reverse_ids(np.array([3, 4]))
        array([-1, 0])
        """
        if assume_valid:
            if self.min_key is None:
                return self.reverse_IDs[ids]
            else:
                return self.reverse_IDs[ids - self.min_key]
        else:
            if self.min_key is None:
                return query_array(self.reverse_IDs, ids)
            else:
                return query_array(self.reverse_IDs, ids - self.min_key)

    def __call__(self, ids, assume_valid=False):
        """Alias for query function."""
        return self.query(ids, assume_valid=assume_valid)

    def query_matched(self, ids, assume_valid=False):
        """Find indices eof the input (external) IDs, also listing matches.

        Parameters
        ----------
        ids : ndarray(int)
            The IDs whose indices in the internal list (used to set up the
            reverse list) should be determined. Out-of-bound situations are
            dealt with internally.
        assume_valid : bool, optional
            Assume that all input IDs are within the range of the internal
            reverse-index list, so that out-of-bound check can be skipped
            (default: ``False``).

        Returns
        -------
        indices : ndarray(int)
            The indices corresponding to each input ID (-1 if not found).
        matches : ndarray(int)
            The indices into the input `ids` array for keys that could be
            matched (i.e. have a non-negative value of `indices`).

        Example
        -------
        >>> import numpy as np
        >>> ids = np.array([4, 0, 5, 1])
        >>> reverse_ids = ReverseList(ids)
        >>> reverse_ids.query_matched(np.array([3, 4]))
        array([-1, 0]), array([1])
        """
        ie_int = self.query(ids, assume_valid=assume_valid)
        ie_matched = np.nonzero(ie_int >= 0)[0]
        return ie_int, ie_matched


def create_reverse_list(ids, delete_ids=False, assume_positive=False,
                        max_val=None):
    """Create a reverse-index list from a (unique) list of IDs.

    This gives the index for each ID value, and is thus the inverse of the
    input list (which gives the ID for an index).

    Warning
    -------
    It is a bad idea to call this function if the input ID list contains very
    large values in relation to the available memory: as a rough guide,
    consider the limit as 1/8 * [RAM/byte]. Ignoring this will result in
    undefined, slow, and likely annoying behaviour.

    Parameters
    ----------
    ids : ndarray (int)
        Input keys (IDs) to invert. These are assumed to be unique;
        death and destruction may happen if this is not the case.
        The array must be one-dimensional. Any negative values are
        assumed to be dummy elements and are ignored.
    delete_ids : bool, optional
        Delete input keys after building reverse list (default: ``False``).
    assume_positive : bool, optional
        Assume that all the input values are non-negative, which speeds
        up the inversion. If ``False`` (default), the code checks
        explicitly which input values are non-negative and only
        transcribes these to the reverse list.
    max_val : int, optional
        Build a reverse list with at least `maxval`+1 entries (i.e. that
        can be indexed by values up to `maxval`), even if this exceeds the
        maximum input ID value. If ``None`` (default), the reverse list is
        built up to self-indexing capacity (i.e. with max(`ids`)+1
        elements).

    Returns
    -------
    rev_IDs : ndarray(int)
        The reverse index list. If the input list contains fewer than
        2 billion elements, it is of type np.int32, otherwise np.int64.

    Note
    ----
    For most practical purposes, it may be more convenient to directly use
    the ReverseList class, or the find_id_indices() function,
    to correlate IDs, both of which call this function internally.

    Example
    -------
    >>> # Standard use to invert an array:
    >>> import numpy as np
    >>> ids = np.array([4, 0, 5, 1])
    >>> create_reverse_list(ids)
    array([1, 3, -1, -1, 0, 2])

    >>> # Including negative array values:
    >>> ids = np.array([4, 0, -1, 5, 1])
    >>> create_reverse_list(ids)
    array([1, 4, -1, -1, 0, 3])

    >>> # Including use of max_val:
    >>> create_reverse_list(ids, maxVal=10)
    array([1, 4, -1, -1, 0, 3, -1, -1, -1, -1, -1])
    """
    # If there is no input, return an empty array
    if len(ids) == 0:
        if max_val is None:
            return np.zeros(0, dtype=np.int32)
        else:
            return np.zeros(max_val+1, dtype=np.int32) - 1

    # Find extent and required data type of output list
    max_ID = ids.max()
    if max_val is not None:
        if max_val > max_ID:
            max_ID = max_val
    if len(ids) > 2e9:
        dtype = np.int64
    else:
        dtype = np.int32

    # Do the actual inversion
    rev_IDs = np.zeros(np.int64(max_ID+1), dtype=dtype) - 1
    if assume_positive:
        rev_IDs[ids] = np.arange(len(ids))
    else:
        ind_good = np.nonzero(ids >= 0)[0]
        rev_IDs[ids[ind_good]] = ind_good

    # Delete input if necessary, to save space
    if delete_ids:
        del ids

    return rev_IDs


def find_id_indices(ids_ext, ids_int, max_direct=int(1e10),
                    sort_below=None, force_sort=False, sort_matches=True,
                    verbose=False):
    """Find and return the locations of IDs in a reference list.

    This function can be used to translate indices in one ('external',
    subject) catalogue into indices in a second ('internal', reference)
    catalogue, for key sets of arbitrary length or values. If the maximum
    value is below an (adjustable) threshold, the lookup is performed
    via an explicit reverse-lookup list. Otherwise, the reference
    list is sorted and the match done via numpy's searchsorted.

    Parameters
    ----------
    ids_ext : ndarray (int)
        Array of keys (IDs) whose indices in `ids_int` should be returned.
        It should be unique, unless the search is guaranteed to be
        executed via a reverse list.
    ids_int : ndarray (int)
        Internal ('reference') list of keys (IDs), assumed to be unique.
        The function will search for each input ID in this array.
    max_direct : int, optional
        Maximum value in either key set for which a reverse-index
        based match is performed, rather than a sort-based search
        (default: 1e10).
    sort_below : int or ``None``, optional
        Maximum length of either ID list for which sort-based search is
        preferred (default: 100). If ``None``, always use reverse-index
        method if possible.
    force_sort : bool, optional
        Force a sorted search irrespective of maximum input values. This
        is equivalent to, but slightly faster than, setting
        `max_direct` = 0.
    sort_matches : bool, optional
        Explicitly sort matching indices from sort-based search in ascending
        order, so that its result is identical to reverse-list based
        method (default: ``True``).
    verbose : bool, optional
        Print timing information (default: ``False``)

    Returns
    -------
    ie_int : ndarray(int)
        The index in ids_int for each input ID (-1 if it could not be
        located at all).
    ie_matched : ndarray(int)
        The input (external) ID indices that could be matched.

    Note
    ----
    For large arrays, using a reverse-lookup list is typically much faster,
    but may use substantial amounts of memory. For small input lists, the
    sort-and-search approach may be faster because it avoids the overheads
    of setting up the reverse list.
    """
    # Determine whether we can use the direct (reverse-ID-list-based) method
    use_direct = True
    if force_sort:
        use_direct = False
    elif sort_below is not None:
        if max(len(ids_ext), len(ids_int)) > sort_below:
            use_direct = False
    elif max(np.max(ids_ext), np.max(ids_int)) > max_direct:
        use_direct = False

    if use_direct:
        # Quick and simple way via reverse ID list:
        rev_IDs = ReverseList(ids_int)
        ie_int = rev_IDs.query(ids_ext)
        ie_matched = np.nonzero(ie_int >= 0)[0]

    else:
        # Need to identify matching IDs on an individual basis.
        time_start = time.time()
        sorter_int = np.argsort(ids_int)
        time_sort = time.time() - time_start
        
        guess = np.searchsorted(
            ids_int, ids_ext, sorter=sorter_int, side='right') - 1
        ie_matched = np.nonzero(ids_int[sorter_int[guess]] == ids_ext)[0]
        ie_int = np.zeros(len(ids_ext), dtype=int) - 1
        ie_int[ie_matched] = sorter_int[guess[ie_matched]]
        
        # If desired, sort matches
        if sort_matches:
            ie_matched = np.sort(ie_matched)

        if verbose:
            time_lookup = time.time() - time_sort
            print("Sorting took    {:.3f} sec." .format(time_sort))
            print("Lookup took     {:.3f} sec." .format(time_lookup))

    return ie_int, ie_matched


def query_array(array, indices, default=-1):
    """Retrieve values from an array that may be too short.

    Parameters
    ----------
    array : ndarray
        The array to retrieve values from. It should be of numerical
        type, and one-dimensional.
    indices : ndarray(int)
        Indices of array to query.
    default : value, optional
        The value to assign to out-of-bound indices (default: -1).

    Returns
    -------
    values : ndarray
        The array values of the specified indices, with out-of-bounds
        indices (negative or >= len(`array`)) set to `default`.

    Example
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 4, 0, 8])
    >>> ind = np.array([0, 2, 8])
    >>> query_array(arr, ind, default=-100)
    array([1, 0, -100])
    """
    if np.min(indices) >= 0 and np.max(indices) < len(array):
        return array[indices]
    else:
        values = np.zeros(len(indices), dtype=array.dtype) + default
        ind_in_range = np.nonzero((indices < len(array)) &
                                  (indices >= 0))[0]
        values[ind_in_range] = array[indices[ind_in_range]]
        return values


def find_vr_haloes(ids, vr_properties_name):
    """Find the VR halo indices belonging to the specified particle IDs."""

    # Need to construct the names of the two files holding the particle
    # linking information.
    vr_group_file = vr_properties_name.replace('properties', 'catalog_groups')
    vr_particle_file = vr_properties_name.replace(
        'properties', 'catalog_particles')
        
    # Load particle offsets and IDs
    with h5.File(vr_group_file, 'r') as f:
        vr_offsets = f['Offset'][...]
        vr_lengths = f['Group_Size'][...]
    with h5.File(vr_particle_file, 'r') as f:
        vr_ids = f['Particle_IDs'][...]

    vr_ends = vr_offsets + vr_lengths

    # Locate 'our' IDs in the VR ID list
    ind_in_vr, found_in_vr = find_id_indices(ids, vr_ids)

    print(f'Located {len(found_in_vr)/len(ind_in_vr)*100:.3f}% of particles '
          f'in VR ID list...')
    
    # Now convert VR particle indices to halo indices
    halo_guess = np.searchsorted(vr_offsets, ind_in_vr[found_in_vr],
                                 side='right') - 1
    ind_good = np.nonzero(
        (ind_in_vr[found_in_vr] >= vr_offsets[halo_guess]) &
        (ind_in_vr[found_in_vr] < vr_ends[halo_guess]))[0]

    print(f'Located {len(ind_good)/len(found_in_vr)*100:.3f}% of particles '
          f'on VR ID list in a halo...')
    
    vr_halo = np.zeros(len(ids), dtype=int) - 1
    vr_halo[found_in_vr[ind_good]] = halo_guess[ind_good]

    return vr_halo
