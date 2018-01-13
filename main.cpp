#include <iostream>
#include <memory>
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Net/Net.h"
int main() {


    std::vector<unsigned short> topology;
    topology.reserve(4);//Number of layers, including input and output layers
    topology.emplace_back(2);//Input layer size
    topology.emplace_back(4);//First layer size
    topology.emplace_back(6);//Second layer size
    topology.emplace_back(1);//Output layer size

    Net *net = new Net(topology);
    std::vector<double> inputVals(2);
    std::vector<double> resultVals(1);
    std::vector<double> targetVals(1);

    //Training Process for an Or gate example
    for(int x = 0; x < 2000; x++)
    {
        //Random Binomial Input Values of either 0 or 1
        inputVals[0] = rand()%2;
        inputVals[1] = rand()%2;
        std::cout<<"First Val: "<<inputVals[0]<<std::endl;
        std::cout<<"Second Val: "<<inputVals[1]<<std::endl;
        net->feedForward(inputVals);

        //Target Value
        targetVals[0] = ((inputVals[0] == 1)|(inputVals[1] == 1) ? 1 : 0);
        std::cout<<"Target Val: "<<targetVals[0]<<std::endl;

        net->backProp(targetVals);//Adjusts Weights
        net->getResults(resultVals);
        //Print out NN's results
        for(int x=0;x<resultVals.size(); x++)
        {
            std::cout<<"Result " << x << ": "<<resultVals[x]<<std::endl<<std::endl;
        }
    }
    //Loops through UI
    while(true)
    {
        std::vector<std::string> s_UserInput(2);
        std::vector<double> UserInput(2);

        //First Value
        std::cout<< "(Enter x to escape) First Val: "<<std::endl;
        getline(std::cin, s_UserInput[0]);//Record line
        assert(s_UserInput[0]!="x");//Escape
        UserInput[0] = atof(s_UserInput[0].c_str());//Convert std::string to double

        //Second Value
        std::cout<<"(Enter x to escape) Second Val: "<<std::endl;
        getline(std::cin, s_UserInput[1]);
        assert(s_UserInput[1]!="x");
        UserInput[1] = atof(s_UserInput[1].c_str());

        net->feedForward(UserInput);//Send Input Values into the Neural Net
        net->getResults(resultVals);//Retrieve results
        std::cout<<resultVals[0]<<std::endl;//Print out results
    }
    return 0;
}
