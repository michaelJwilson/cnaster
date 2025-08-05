# TODO validate
def cast_csr(csr_matrix):
    result = []

    for i in range(csr_matrix.shape[0]):
        start_idx = csr_matrix.indptr[i]
        end_idx = csr_matrix.indptr[i + 1]

        row_data = []

        for idx in range(start_idx, end_idx):
            col = csr_matrix.indices[idx]
            val = csr_matrix.data[idx]
            row_data.append((col, val))

        result.append(row_data)

    return result