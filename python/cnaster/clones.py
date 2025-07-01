def perform_partition(coords, sample_ids, x_part, y_part, single_tumor_prop, threshold):
    initial_clone_index = []
    
    for s in range(np.max(sample_ids) + 1):
        index = np.where(sample_ids == s)[0]
        assert len(index) > 0
        if single_tumor_prop is None:
            tmp_clone_index = fixed_rectangle_initialization(
                coords[index, :], x_part, y_part
            )
        else:
            tmp_clone_index = fixed_rectangle_initialization_mix(
                coords[index, :],
                x_part,
                y_part,
                single_tumor_prop[index],
                threshold=threshold,
            )
        for x in tmp_clone_index:
            initial_clone_index.append(index[x])
    return initial_clone_index
