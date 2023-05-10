#pragma once
#include <vector>
#include <map>
#include <string>

using std::vector;
using std::map;
using std::string;


class DecisionTree {
    private:
        vector <int> child_left;
        vector <int> child_right;
        vector <int> features;
        vector <double> thresholds;
        vector <map<string, string>> values;

        bool is_leaf(int node);

        int next_node(int node, vector<double> &state);
    public:
        DecisionTree(string filepath);
        map<string, string> predict(vector<double> &state);
};