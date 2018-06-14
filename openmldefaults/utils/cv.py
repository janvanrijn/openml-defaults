
def get_cv_indices(num_tasks, num_iteration, iteration):
    lower_bound = (num_tasks / num_iteration) * iteration
    upper_bound = (num_tasks / num_iteration) * (iteration + 1)

    holdout_set = [i for i in range(num_tasks) if lower_bound <= i < upper_bound]
    return holdout_set
