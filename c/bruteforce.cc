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

#include <iostream>

using namespace std;

double const MAX_LOSS = 1.0;
int const MAX_NUM_TASKS = 30;
int const MAX_NUM_CONFIGS = 500;
int const MAX_NUM_DEFAULTS = 10;

float current_best_solution;


void print_subset(int num_defaults, int current_subset[MAX_NUM_DEFAULTS]) {
  for (int i = 0; i < num_defaults - 1; ++i) {
    cout << current_subset[i] << ",";
  }
  cout << current_subset[num_defaults-1];
}


double evaluate_subset(int num_tasks, int num_defaults, 
                       double performance[MAX_NUM_CONFIGS][MAX_NUM_TASKS],
                       int current_subset[MAX_NUM_DEFAULTS]) {
  double sum_of_losses = 0;
  for (int i = 0; i < num_tasks; ++i) {
    double current_task_loss = MAX_LOSS;
    for (int j = 0; j < num_defaults; ++j) {
      double current_config_score = performance[i][current_subset[j]];
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
  if (current_index >= num_defaults) {
    double current_score = evaluate_subset(num_tasks, num_defaults, performance,
                                           current_subset);
    
    if (current_score < current_best_solution) {
      cout << "Found new solution: " << endl;
      current_best_solution = current_score;
      print_subset(num_defaults, current_subset);
      cout << endl << current_score << endl;
    }
  } else {
    int start_index = 0;
    if (current_index > 0) {
      start_index = current_subset[current_index - 1] + 1;
    }
    
    for (int i = start_index; i < num_configurations; ++i) {
      current_subset[current_index] = i;
      generate_subsets(num_tasks, num_configurations, num_defaults, 
                       performance, current_index + 1, current_subset);
    }
  }
}


int main() {
  int num_defaults;
  int num_configs;
  int num_tasks;
  double performance[MAX_NUM_CONFIGS][MAX_NUM_TASKS];
  int current_subset[MAX_NUM_DEFAULTS];
  
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
    for (int j = 0; j < num_configs; ++j) {
      cin >> performance[i][j];
    }
  }
  
  current_best_solution = num_tasks * MAX_LOSS;  
  generate_subsets(num_tasks, num_configs, num_defaults, performance,
                   0, current_subset);
}

