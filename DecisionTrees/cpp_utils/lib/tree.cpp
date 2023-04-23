#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <sstream>
#include "tree.hpp"

#include <iostream>

using std::vector;
using std::string;
using std::map;
using std::ifstream;
using std::stringstream;
using std::stoi;


template<class T>
vector<T> stovec(string s) {
    stringstream iss(s);
    T el; vector<T> elements;
    while (iss >> el) {
        elements.push_back(el);
    }
    return elements;
}


bool DecisionTree::is_leaf(int node) {
    return child_left[node] == child_right[node];
}

int DecisionTree::next_node(int node, vector<double> &state) {
    if (thresholds[node] < state[features[node]]) {
        return child_right[node];
    } else {
        return child_left[node];
    }
}

DecisionTree::DecisionTree(string filepath) {
    ifstream input(filepath);
    string line;

    getline(input, line); 
    assert(line == "Node count");
    getline(input, line);
    int node_count = stoi(line);

    getline(input, line);
    assert(line == "Children left");
    getline(input, line);
    this->child_left = stovec<int>(line);

    getline(input, line);
    assert(line == "Children right");
    getline(input, line);
    this->child_right = stovec<int>(line);

    getline(input, line);
    assert(line == "Feature");
    getline(input, line);
    this->features = stovec<int>(line);

    getline(input, line);
    assert(line == "Threshold");
    getline(input, line);
    this->thresholds = stovec<double>(line);

    getline(input, line);
    assert(line == "Values");

    vector<map<string, string>> vec(node_count);
    this->values = vec;
    for (int i = 0; i < node_count; i++) {
        map<string, string> p_map;
        this->values[i] = p_map;
    }

    while (input.peek() != EOF) {
        string p_name;
        getline(input, p_name);
        for (int cur_node = 0; cur_node < node_count; cur_node++) {
            string p_value;
            getline(input, p_value);
            this->values[cur_node][p_name] = p_value;
        }
    }
}

map<string, string> DecisionTree::predict(vector<double> &state) {
    int cur_node = 0;
    while (!is_leaf(cur_node)) {
        cur_node = next_node(cur_node, state);
    }
    return values[cur_node];
}
