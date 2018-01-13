cmake_minimum_required(VERSION 3.9)
project(NeuralOrGate)

set(CMAKE_CXX_STANDARD 11)

add_executable(NeuralOrGate main.cpp NeuralNetwork/Neuron/Neuron.h NeuralNetwork/Net/Net.h NeuralNetwork/Neuron/Neuron.cpp NeuralNetwork/Net/Net.cpp)