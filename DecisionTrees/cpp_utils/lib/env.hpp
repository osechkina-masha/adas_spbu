#ifndef decision_cpp_env
#define decision_cpp_env

#include <string>
#include <vector>
#include <map>

using std::string;
using std::vector;
using std::map;

class Parameter {
    private:
        string name;
    public:
        virtual string serialize() = 0;
        virtual void set_name(string name);
        virtual string get_name();
};

class DiscreteParameter: public Parameter {
    private:
        vector<string> values;
    public:
        DiscreteParameter(vector<string> values);
        string serialize();
};


class ContinuousParameter: public Parameter {
    private:
        double min_value; 
        double max_value;
    public:
        ContinuousParameter(double min_value, double max_value);
        string serialize();
};


class ParametersDescription {
    private:
        vector<ContinuousParameter> continuous;
        vector<DiscreteParameter> discrete;
    public:
        ParametersDescription* add_continuous(double min_value, double max_value);
        ParametersDescription* add_discrete(vector<string> values);
        vector<DiscreteParameter> get_discrete();
        vector<ContinuousParameter> get_continuous();
};


class CppEnvironment {
    private:
        vector<DiscreteParameter> discrete_parameters_desc;
        vector<ContinuousParameter> continuous_parameters_desc;
    public:
        CppEnvironment(ParametersDescription param_desc); 

        virtual double score(map<string, string> &parameters) = 0;
        virtual void reset() = 0;
        virtual vector<double> current_state() = 0;

        void communicate();
};

#endif