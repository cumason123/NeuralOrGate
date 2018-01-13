//
// Created by Curtis Mason on 1/12/18.
//
#ifndef VECTOR
#define VECTOR
#include <vector>
#endif

#ifndef NEURALORGATE_NEURON_H
#include "../Neuron/Neuron.h"
#endif

#ifndef NEURALORGATE_NET_H
#define NEURALORGATE_NET_H

#ifndef CMATH
#define CMATH
#include <cmath>
#endif

#ifndef CASSERT
#define CASSERT
#include "cassert"
#endif

class Net
{
public:
    Net(const std::vector<unsigned short> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals);
private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum], the entire neural net
    double m_error;
    double m_recentAverageSmoothingFactor;
    double m_recentAverageError;
};







#endif //NEURALORGATE_NET_H
