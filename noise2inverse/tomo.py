import astra
import numpy as np
import torch
from tqdm import tqdm
import tomosipo as ts


def filter_in_real_filterspace(n):
    # Makes filter in real filter space.
    # Complex component equals zero.
    filter = np.zeros(n)
    filter[0] = 0.25
    # even indices are zero
    # for odd indices j, filter[j] equals
    #   -1 / (pi * j) ** 2,          when 2 * j <= n
    #   -1 / (pi * (n - j)) ** 2,    when 2 * j >  n

    odd_indices = np.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filter[1::2] = -1 / (np.pi * odd_indices) ** 2

    return filter


def filter_proj_data(sino):
    """Filters projection data for FBP

    Uses Ram-Lak filter.

    Follows the approach of:

    Zeng GL. Revisit of the Ramp Filter. IEEE Trans Nucl Sci. 2015;62(1):131–136. doi:10.1109/TNS.2014.2363776

    :param sino: `torch.tensor` projection data
    :returns: `torch.tensor` filtered projection data
    :rtype:

    """
    normalized = False
    (num_slices, num_angles, num_pixels) = sino.shape
    # We must have:
    # 1) num_pixels + num_padding is even (because rfft wants the #input elements to be even)
    # 2) num_padding // 2 must equal at least num_pixels (not so sure about this actually..)
    if num_pixels % 2 == 0:
        num_padding = num_pixels
    else:
        num_padding = num_pixels + 2

    # num_padding < num_padding_left
    num_padding_left = num_padding // 2

    # M is always even
    M = num_pixels + num_padding

    tmp_sino = sino.new_zeros((num_slices, num_angles, M))
    tmp_sino[:, :, num_padding_left:num_padding_left + num_pixels] = sino

    # XXX: Consider using torch.stft. This might save us from doing
    # the padding.
    fourier_sino = torch.rfft(
        tmp_sino,
        signal_ndim=1,
        normalized=normalized
    )

    real_filter = filter_in_real_filterspace(M).astype(np.float32)
    fourier_filter = torch.rfft(
        torch.from_numpy(real_filter),
        signal_ndim=1,
        normalized=normalized,
    )
    # Make complex dimension equal to real dimension
    fourier_filter = fourier_filter[:, 0][:, None]

    fourier_sino *= fourier_filter
    tmp_filtered = torch.irfft(
        fourier_sino,
        signal_ndim=1,
        signal_sizes=(M,),
        normalized=normalized,
    )
    tmp_filtered /= num_angles / np.pi

    filtered = tmp_filtered.new_empty(sino.shape)
    filtered[:] = tmp_filtered[:, :, num_padding_left:num_padding_left + num_pixels]

    return filtered


def fbp(A, sino):
    # Filter
    filtered = filter_proj_data(
        torch.from_numpy(sino),
    ).detach().numpy()
    # Reconstruct
    return A.T(filtered)
