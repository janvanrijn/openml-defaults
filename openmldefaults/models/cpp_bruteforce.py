import json
import openmldefaults
import subprocess
import time


class CppDefaults(object):

    def __init__(self, c_executable, apply_branch_bound):
        self.c_executable = c_executable
        self.name = 'cpp_bruteforce'
        self.apply_branch_bound = apply_branch_bound

    def generate_defaults(self, df, num_defaults):
        num_configs, num_tasks = df.shape
        print(openmldefaults.utils.get_time(), 'Started c program')
        process = subprocess.Popen([self.c_executable], stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if self.apply_branch_bound:
            branch_and_bound = "1"
        else:
            branch_and_bound = "0"
        process_input = [branch_and_bound, str(num_defaults), str(num_configs), str(num_tasks)]
        for iConf in range(num_configs):
            for iTask in range(num_tasks):
                process_input.append(str(df.iloc[iConf, iTask]))

        start_time = time.time()
        out, err = process.communicate("\n".join(process_input))
        runtime = time.time() - start_time
        if process.returncode != 0:
            raise ValueError('Process terminated with non-zero exit code. Stderr: ' + err)
        print(openmldefaults.utils.get_time(), 'Runtime: %d seconds' % runtime)

        for idx, line in enumerate(out.split("\n")):
            try:
                solution = json.loads(line)
            except json.decoder.JSONDecodeError:
                raise ValueError('Could not parse result as json: %s' % line)

        print(openmldefaults.utils.get_time(), solution)
        selected_defaults = [df.index[idx] for idx in solution['solution']]

        results_dict = {
            'objective': solution['score'],
            'run_time': runtime,
            'defaults': selected_defaults,
            'nodes_visited': solution['nodes_visited'],
            'leafs_visited': solution['leafs_visited'],
            'branch_and_bound': solution['branch_and_bound']
        }
        return results_dict
