//
// Created by Curtis Mason on 1/12/18.
//
#ifndef VECTOR
#define VECTOR
#include <vector>
#endif

#ifndef CMATH
#define CMATH
#include <cmath>
#endif

#ifndef NEURALORGATE_NEURON_H
#define NEURALORGATE_NEURON_H

struct Connection
{
    double weight;
    unsigned deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(Layer &prevLayer);
    void setOutputVal(double val);
    double getOutputVal(void);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double transferFunctionDerivative(double x);
    static double transferFunction(double x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer)const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights; //weigh value for outputs
    unsigned m_myIndex;
    double m_gradient;
    static double eta; // [0.0, 1.0] overall net training weight
    static double alpha;// [0.0, n] Multiplier of last weight change (momentum)
};


#endif //NEURALORGATE_NEURON_H
