//
// Created by Curtis Mason on 1/12/18.
//
#include "Neuron.h"
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    //numOutputs is the number of neurons in the next layer excluding bias neuron
    //I don't have an explanation for myIndex atm tbh

    //Appends random weights upon initialization of the neuron
    for (unsigned c=0; c<numOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}
double Neuron::randomWeight()
{
    return rand()/double(RAND_MAX);
}
void Neuron::setOutputVal(double val)
{
    m_outputVal = val;
}
double Neuron::getOutputVal(void)
{
    return m_outputVal;
}
void Neuron::feedForward(Layer &prevLayer)
{
    double sum = 0.0;
    //Sum the previous layer's outputs which are our inputs
    //include the bias node from the previous layer, and multiply them by weights
    for(unsigned n = 0; n< prevLayer.size(); n++)
    {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    //Real life application of a sigmoid curve, WOW math
    m_outputVal = Neuron::transferFunction(sum);
}


void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}


double Neuron::sumDOW(const Layer &nextLayer)const
{
    double sum = 0.0;
    //Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}


void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}


double Neuron::transferFunction(double x)
{
    //tanh - output range [-1.0, 1.0]
    return tanh(x);
}


double Neuron::transferFunctionDerivative(double x)
{
    //PS, tanh is a hyperbolic function, the more you know, and
    // sech^(x) is the derivative of tanh(x)
    return (1/cosh(x))*(1/cosh(x));
}


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


void Neuron::updateInputWeights(Layer &prevLayer)
{
    //The weights to be updated are in the COnections container
    //In the neurons in the preceding layer
    for (unsigned n = 0; n<prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight =
                //Individual input, magnified by the gradient and train rate;
                eta //overall net learning rate
                * neuron.getOutputVal()
                * m_gradient
                // also add momentum = a function of the previous delta weight
                + alpha //momentum
                  * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

    }
}