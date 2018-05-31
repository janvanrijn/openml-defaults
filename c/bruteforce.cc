/*
  Enumerates all subsets of the configurations (subset of size num_defaults) to
  determine what is the best set of complementary defaults. 
  
  Input format:
  int (num defaults)
  int (num configurations)
  int (num tasks)
  for every configuration c
    for every task t
      double (the score of configuration c on task t)
*/

#include <cmath>
#include <iostream>

using namespace std;

bool branch_and_bound;
double const MAX_LOSS = 1.0;
int const MAX_NUM_TASKS = 30;
int const MAX_NUM_CONFIGS = 500;
int const MAX_NUM_DEFAULTS = 10;
int nodes_visited = 0;

float current_best_solution = MAX_NUM_TASKS * MAX_LOSS;

// matrix with c columns and r rows. Each cell (c, r) represents the best 
// obtainable score on task r with configurations c - MAX_NUM_CONFIGS
float dp_best_obtainable[MAX_NUM_CONFIGS][MAX_NUM_TASKS];


void print_array(int array_size, int array[]) {
  for (int i = 0; i < array_size - 1; ++i) {
    cout << array[i] << ",";
  }
  cout << array[array_size-1];
}

void print_array_d(int array_size, double array[]) {
  for (int i = 0; i < array_size - 1; ++i) {
    cout << array[i] << ",";
  }
  cout << array[array_size-1];
}

double evaluate_subset(int num_tasks, int num_defaults, 
                       double performance[MAX_NUM_CONFIGS][MAX_NUM_TASKS],
                       int current_subset[MAX_NUM_DEFAULTS]) {
  double sum_of_losses = 0;
  for (int i = 0; i < num_tasks; ++i) {
    double current_task_loss = MAX_LOSS;
    for (int j = 0; j < num_defaults; ++j) {
      double current_config_score = performance[current_subset[j]][i];
      if (current_config_score < current_task_loss) {
        current_task_loss = current_config_score;
      }
    }
    sum_of_losses += current_task_loss;
  }
  return sum_of_losses;
}


void generate_subsets(int num_tasks, int num_configurations, int num_defaults,
                      double performance[MAX_NUM_CONFIGS][MAX_NUM_TASKS],
                      int current_index, int current_subset[MAX_NUM_DEFAULTS]) {
  nodes_visited += 1;
  
  if (current_index >= num_defaults) {
    double current_score = evaluate_subset(num_tasks, num_defaults, performance,
                                           current_subset);
    
    if (current_score < current_best_solution) {
      current_best_solution = current_score;
      cout << "{";
      cout << "\"solution\": [";
      print_array(num_defaults, current_subset);
      cout << "]";
      cout << ", \"score\": " << current_score;
      cout << ", \"branch_and_bound\": " << branch_and_bound ;
      cout << ", \"nodes_visited\": " << nodes_visited;
      //for (int i = 0; i < num_defaults; ++i) {
      //  cout << ", \"losses_config_" << current_subset[i] << "\": [";
      //  print_array_d(num_tasks, performance[current_subset[i]]);
      //  cout << "]";
      //}
      cout << "}" << endl;
    }
  } else {
    int start_index = 0;
    if (current_index > 0) {
      start_index = current_subset[current_index - 1] + 1;
    }
    
    // calculate the potential (or we will cut off the branch)
    if (branch_and_bound) {
      float sum_best_obtainable = 0;
      for (int i = 0; i < num_tasks; ++i) {
        float best_obtainable_task = dp_best_obtainable[start_index][i];
        for (int j = 0; j < current_index; ++j) {
          best_obtainable_task = fmin(performance[current_subset[j]][i], 
                                      best_obtainable_task);
        }
        sum_best_obtainable += best_obtainable_task;
      }
      if (sum_best_obtainable > current_best_solution) {
        return;
      }
    }
    
    for (int i = start_index; i < num_configurations; ++i) {
      current_subset[current_index] = i;
      generate_subsets(num_tasks, num_configurations, num_defaults, 
                       performance, current_index + 1, current_subset);
    }
  }
}

void fill_dp(int num_tasks, int num_configurations, 
             double performance[MAX_NUM_CONFIGS][MAX_NUM_TASKS]) {
  for (int i = 0; i < num_tasks; ++i) {
    float current_min = MAX_LOSS;
    for (int j = num_configurations - 1; j >= 0; --j) {
      current_min = fmin(current_min, performance[j][i]);
      dp_best_obtainable[j][i] = current_min;
    }
  }
}


int main() {
  int num_defaults;
  int num_configs;
  int num_tasks;
  double performance[MAX_NUM_CONFIGS][MAX_NUM_TASKS];
  int current_subset[MAX_NUM_DEFAULTS];
  
  cin >> branch_and_bound;
  cin >> num_defaults;
  if (num_defaults > MAX_NUM_DEFAULTS) {
    cout << "Num defaults too high. Max allowed: " << MAX_NUM_DEFAULTS << endl;
    return 1;
  }
  cin >> num_configs;
  if (num_configs > MAX_NUM_CONFIGS) {
    cout << "Num configs too high. Max allowed: " << MAX_NUM_CONFIGS << endl;
    return 1;
  }
  cin >> num_tasks;
  if (num_tasks > MAX_NUM_TASKS) {
    cout << "Num tasks too high. Max allowed: " << MAX_NUM_TASKS << endl;
    return 1;
  }
  
  for (int i = 0; i < num_configs; ++i) {
    for (int j = 0; j < num_tasks; ++j) {
      cin >> performance[i][j];
    }
  }
  fill_dp(num_tasks, num_configs, performance);
  current_best_solution = num_tasks * MAX_LOSS;  
  generate_subsets(num_tasks, num_configs, num_defaults, performance,
                   0, current_subset);
}

