import pandas as pd
from typing import Callable


def dominates_min(dominater, dominated):
    return sum([dominater[x] <= dominated[x] for x in range(len(dominater))]) == len(dominater)


# fn from: http://code.activestate.com/recipes/578287-multidimensional-pareto-front/
def simple_cull(original_frame: pd.DataFrame, dominates: Callable):
    len_original_frame = len(original_frame)
    # copies the frame. contains the same number of rows (filled with na's) should be removed later
    pareto_frame = pd.DataFrame(data=None, columns=original_frame.columns, index=original_frame.index)
    dominated_frame = pd.DataFrame(data=None, columns=original_frame.columns, index=original_frame.index)
    candidate_row_nr = 0

    while True:
        # obtain the current row
        candidate_key = original_frame.index[candidate_row_nr]
        candidate_row = original_frame.loc[candidate_key]
        # remove it from the original frame
        original_frame = original_frame.drop(candidate_key, axis=0)

        # iterate over other rows
        alternative_row_nr = 0
        non_dominated = True
        while len(original_frame) != 0 and alternative_row_nr < len(original_frame):
            alternative_key = original_frame.index[alternative_row_nr]
            alternative_row = original_frame.loc[alternative_key]
            if dominates(candidate_row, alternative_row):
                # Candidate row is dominated so remove it from the array
                original_frame = original_frame.drop(alternative_key, axis=0)
                dominated_frame.loc[alternative_key] = alternative_row
            elif dominates(alternative_row, candidate_row):
                non_dominated = False
                dominated_frame.loc[candidate_key] = candidate_row
                alternative_row_nr += 1
            else:
                alternative_row_nr += 1

        if non_dominated:
            # add the non-dominated point to the Pareto frontier
            pareto_frame.loc[candidate_key] = candidate_row

        if len(original_frame) == 0:
            break

    pareto_frame = pareto_frame.dropna()
    dominated_frame = dominated_frame.dropna()

    if len(pareto_frame) + len(dominated_frame) != len_original_frame:
        raise ValueError('Original frame has %d rows. Pareto %d, Dominated %d' % (len_original_frame,
                                                                                  len(pareto_frame),
                                                                                  len(dominated_frame)))

    return pareto_frame, dominated_frame
