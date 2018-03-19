import numpy as np

def calc_entropy_for_specipic_t(current_ts, px_i):
	"""Calc entropy for specipic t"""
	b2 = np.ascontiguousarray(current_ts).view(
		np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_current_ts = unique_counts / float(sum(unique_counts))
	p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
	H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
	return H2X

def calc_condtion_entropy(px, t_data, unique_inverse_x):
	# Condition entropy of t given x
	H2X_array = np.array([calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i]) for i in range(px.shape[0])])
	H2X = np.sum(H2X_array)
	return H2X

def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array):
    """Calculate the MI based on binning of the data"""
    H2 = -np.sum(ps2 * np.log2(ps2))
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
    IY = H2 - H2Y
    IX = H2 - H2X
    return IX, IY

def extract_probs(label, x):
	"""calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
	pys = np.sum(label, axis=0) / float(label.shape[0])
	b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
	unique_array, unique_indices, unique_inverse_x, unique_counts = \
		np.unique(b, return_index=True, return_inverse=True, return_counts=True)
	unique_a = x[unique_indices]
	b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
	pxs = unique_counts / float(np.sum(unique_counts))
	p_y_given_x = []
	for i in range(0, len(unique_array)):
		indexs = unique_inverse_x == i
		py_x_current = np.mean(label[indexs, :], axis=0)
		p_y_given_x.append(py_x_current)
	p_y_given_x = np.array(p_y_given_x).T
	b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
	unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
		np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
	pys1 = unique_counts_y / float(np.sum(unique_counts_y))
	return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs

def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y, calc_DKL=False):
	bins = bins.astype(np.float32)
	num_of_bins = bins.shape[0]
	# bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
	# hist, bin_edges = np.histogram(np.squeeze(data.reshape(1, -1)), normed=True)
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
	if calc_DKL:
		pxy_given_T = np.array(
			[calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
		)
		p_XgT = np.vstack(pxy_given_T[:, 0])
		p_YgT = pxy_given_T[:, 1]
		p_YgT = np.vstack(p_YgT).T
		DKL_YgX_YgT = np.sum([inf_ut.KL(c_p_YgX, p_YgT.T) for c_p_YgX in p_YgX.T], axis=0)
		H_Xgt = np.nansum(p_XgT * np.log2(p_XgT), axis=1)
	local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
	                                                 unique_array)
	return local_IXT, local_ITY

def calc_information_for_layer_with_other(data, bins, unique_inverse_x, unique_inverse_y, label,
                                          b, b1, len_unique_a, pxs, p_YgX, pys1,
                                          percent_of_sampling=50):
	local_IXT, local_ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
	                                                 len_unique_a, p_YgX, unique_inverse_x,
	                                                 unique_inverse_y)
	params = {}
	params['local_IXT'] = local_IXT
	params['local_ITY'] = local_ITY
	return params