#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <sstream>
#include "env.hpp"

using std::vector;
using std::string;
using std::map;
using std::ifstream;
using std::stringstream;

// Parameter
void Parameter::set_name(string name) {
    this->name = name;
}

string Parameter::get_name() {
    return this->name;
}

// Discrete parameter
DiscreteParameter::DiscreteParameter(vector<string> values) {
    this->values = values;
}

string DiscreteParameter::serialize() {
    std::stringstream ss;
    ss << this->get_name() << std::endl;
    for (string v: values) {
        ss << v << std::endl;
    }
    return ss.str();
}

// ContinuousParameters
ContinuousParameter::ContinuousParameter(double min_value, double max_value) {
    this->min_value = min_value;
    this->max_value = max_value;
}

string ContinuousParameter::serialize() {
    std::stringstream ss;
    ss << this->get_name() << std::endl;
    ss << this->min_value << std::endl;
    ss << this->max_value << std::endl;
    return ss.str();
}


// Parameters description
ParametersDescription* ParametersDescription::add_continuous(string name, double min_value, double max_value) {
    ContinuousParameter param(min_value, max_value);
    param.set_name(name);
    this->continuous.push_back(param);
    return this;
}

ParametersDescription* ParametersDescription::add_discrete(string name, vector<string> values) {
    DiscreteParameter param(values);
    param.set_name(name);
    this->discrete.push_back(param);
    return this;
}

vector<DiscreteParameter> ParametersDescription::get_discrete() {
    return this->discrete;
}

vector<ContinuousParameter> ParametersDescription::get_continuous() {
    return this->continuous;
}

// CppEnvirnment
CppEnvironment::CppEnvironment(ParametersDescription param_desc) {
    this->discrete_parameters_desc = param_desc.get_discrete();
    this->continuous_parameters_desc = param_desc.get_continuous();
}

void CppEnvironment::communicate() {
    // Send info about parameters
    std::cout << "N_discrete" << std::endl << this->discrete_parameters_desc.size() << std::endl;
    for (DiscreteParameter p: this->discrete_parameters_desc) {
        std::cout << p.serialize();
    }

    std::cout << "N_continuous" << std::endl << this->continuous_parameters_desc.size() << std::endl;
    for (ContinuousParameter p: this->continuous_parameters_desc) {
        std::cout << p.serialize();
    }

    // Wait for commands
    string command;
    int n_params = this->discrete_parameters_desc.size() + this->continuous_parameters_desc.size();
    while (command != "END") {
        std::getline(std::cin, command);
        switch(command[0]) {
            case 'S':
            {
                map<string, string> parameters;
                for (int i = 0; i < n_params; i++) {
                    string p_name; string p_value;
                    getline(std::cin, p_name);
                    getline(std::cin, p_value);
                    parameters[p_name] = p_value;
                }
                std::cout << this->score(parameters) << std::endl;
                break;
            }
            case 'R':
            {
                this->reset();
                break;
            }
            case 'C':
            {
                vector<double> state = this->current_state();
                for (double el: state) {
                    std::cout << el << std::endl;
                }
                break;
            }
        }
    }
}

