import pandas as pd
from typing import Callable


def dominates(dominater, dominated, original_index_keys=None):
    if dominater.index.names != dominated.index.names:
        raise ValueError()
    if original_index_keys is None:
        original_index_keys = []
    total = 0
    for key in dominater.keys():
        if dominater[key] >= dominated[key] and key not in original_index_keys:
            total += 1
    return total == len(dominater) - len(original_index_keys)


def dominates_min(dominater, dominated):
    return sum([dominater[x] <= dominated[x] for x in range(len(dominater))]) == len(dominater)


# fn from: http://code.activestate.com/recipes/578287-multidimensional-pareto-front/
def simple_cull(original_frame: pd.DataFrame, dominates: Callable):
    original_frame = pd.DataFrame.copy(original_frame)
    original_keys = original_frame.index.names
    if original_frame.index.names == [None]:
        raise ValueError('Frame does not have valid index')
    original_frame.reset_index(inplace=True)

    len_original_frame = len(original_frame)
    # copies the frame. contains the same number of rows (filled with na's) should be removed later
    pareto_frame = pd.DataFrame(data=None, columns=original_frame.columns, index=original_frame.index)
    dominated_frame = pd.DataFrame(data=None, columns=original_frame.columns, index=original_frame.index)
    candidate_row_nr = 0

    while True:
        # obtain the current row
        candidate_row = original_frame.iloc[candidate_row_nr]
        # remove it from the original frame
        original_frame = original_frame.drop(original_frame.index[candidate_row_nr], axis=0)

        # iterate over other rows
        alternative_row_nr = 0
        non_dominated = True
        while len(original_frame) != 0 and alternative_row_nr < len(original_frame):
            alternative_row = original_frame.iloc[alternative_row_nr]
            if dominates(candidate_row, alternative_row, original_keys):
                # Candidate row is dominated so remove it from the array
                original_frame = original_frame.drop(alternative_row_nr, axis=0)
                dominated_frame.iloc[alternative_row_nr] = alternative_row

            elif dominates(alternative_row, candidate_row, original_keys):
                non_dominated = False
                dominated_frame.iloc[candidate_row_nr] = candidate_row
                alternative_row_nr += 1
            else:
                alternative_row_nr += 1

        if non_dominated:
            # add the non-dominated point to the Pareto frontier
            pareto_frame.loc[alternative_row_nr] = candidate_row

        if len(original_frame) == 0:
            break

    pareto_frame = pareto_frame.set_index(original_keys).dropna()
    dominated_frame = dominated_frame.set_index(original_keys).dropna()
    if len(pareto_frame) + len(dominated_frame) != len_original_frame:
        raise ValueError('Original frame has %d rows. Pareto %d, Dominated %d' % (len_original_frame,
                                                                                  len(pareto_frame),
                                                                                  len(dominated_frame)))

    return pareto_frame, dominated_frame
