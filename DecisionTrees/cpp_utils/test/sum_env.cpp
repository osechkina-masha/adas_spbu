#include <iostream>
#include <random>
#include <ctime>
#include "../lib/env.hpp"


class SumEnvironment: public CppEnvironment {
    private:
        int a;
        int b;
    public:
        using CppEnvironment::CppEnvironment;

        double score(map<string, string> &parameters) {
            double sum = stod(parameters["sum"]);
            return -std::abs(sum - this->a - this->b) / 6.0;
        }
        void reset() {
            this->a = std::rand() % 4;
            this->b = std::rand() % 4;
        }
        vector<double> current_state() {
            vector<double> state_vec = {0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0};
            state_vec[this->a] = 1.0;
            state_vec[this->b + 4] = 1.0;
            return state_vec;
        }
};


int main() {
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    ParametersDescription desc;
    desc.add_continuous("sum", 0, 4);
    SumEnvironment env(desc);
    env.communicate();
}   