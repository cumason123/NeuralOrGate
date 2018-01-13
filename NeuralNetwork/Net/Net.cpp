//
// Created by Curtis Mason on 1/12/18.
//
#include "Net.h"
Net::Net(const std::vector<unsigned short> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
    {
        m_layers.push_back(Layer());
        //Appends objects of Layers to the vector m_layers

        unsigned numOutputs =
                layerNum == topology.size() -1 ? 0 : topology[layerNum +1];
        //layerNum+1 equals the num of layers in the next layer

        //Now we must append neurons to said layer
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)//less than equal to to incorporate bias neuron
        {
            //numOutputs is the number of neurons in the next layer, neuronNum = myIndex
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));//most recent element has neurons appended onto it contained
        }
    }
}


void Net::feedForward(const std::vector<double> &inputVals)
{
    //Exception Handler, inputs equal neurons in first layer
    assert(inputVals.size() == m_layers[0].size()-1);

    //Assign (latch) the input values into the input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    //loop through each layer, and each neuron then feed forward

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum -1];
        for(unsigned n = 0; n<m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}


void Net::backProp(const std::vector<double> &targetVals)
{
    //Calculate overall net error (RMS OF OUTPUT NEURON ERRORS)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    //Solve for error per neuron in prev layer
    for(unsigned n = 0; n< outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n]*outputLayer[n].getOutputVal();
        m_error += delta*delta;
    }

    m_error /=outputLayer.size() - 1;//Get average error squared
    m_error = sqrt(m_error);//RMS

    //Implement a recent average measure
    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor+1.0);

    //Calculate output layer gradients
    for (unsigned n = 0; n<outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    //Calculate gradients on hidden layers
    for(unsigned layerNum = m_layers.size()-2; layerNum > 0; --layerNum)
    {
        Layer &currentHiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum +1];

        //calcHiddenGradients for every neuron in the current hidden layer
        for(unsigned n = 0; n<currentHiddenLayer.size(); ++n)
        {
            currentHiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //For all layers from outputs to first hidden layer,
    //update connection weights.
    for(unsigned layerNum = m_layers.size() -1; layerNum>0; --layerNum)
    {
        Layer &currentLayer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        //updatesInputWeights for all neuron's in current layer
        for (unsigned n = 0; n < currentLayer.size() - 1; ++n)
        {
            currentLayer[n].updateInputWeights(prevLayer);
        }
    }

}


void Net::getResults(std::vector<double> &resultVals)
{
    resultVals.clear();
    //cout<<"m_layers.back().size() "<<m_layers.back().size()<<endl;
    for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        //Binds outputValues of all the neurons into the resultVals vector
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

