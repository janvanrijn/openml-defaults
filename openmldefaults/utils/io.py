
def get_experiment_dir(args):
    if args.restricted_num_tasks is not None:
        experiment_dir = 'c%d_t%d_d%d' % (args.resized_grid_size, args.restricted_num_tasks, args.num_defaults)
    else:
        experiment_dir = 'c%d_tAll_d%d' % (args.resized_grid_size, args.num_defaults)
    return experiment_dir