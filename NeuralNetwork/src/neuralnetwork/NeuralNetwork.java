/*
 * 
 * 
 * 
 */
package neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Bruce
 */
public class NeuralNetwork
{
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        // Variables (STILL HARDCODED $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$)
        int numLayers = 2;
        int numNodes;
        int total;
        double seventy;
        int theRest;
        int countRight = 0;
        Instances trainingData;
        Instances testingData;
        Instances data = null;
        DataSource source;
        List<Layer> layers;
        double[] set;
        double rightAnswer;

        // Get the data into a variable Instance
        try
        {
            source = new ConverterUtils.DataSource("C:\\Users\\Bruce\\Documents\\School Spring 2015\\CS450 Mining\\iris.csv");
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            // Randomize the data            
        } catch (Exception e)
        {
            System.out.println("Exception caught: " + e);
        }

        // Makes a list of layers and the correct number of nodes
        layers = Collections.synchronizedList(new ArrayList());
        for (int i = 0; i < numLayers; i++)
        {
            layers.add(new Layer());

            // Set the number of nodes according to the layer we're on (STILL HARDCODED $$$$$$$$)
            if (i == 0)
            {
                numNodes = 4;
            } else
            {
                numNodes = 3;
            }
            for (int j = 0; j < numNodes; j++)
            {
                layers.get(i).addNode(new Node(data.numAttributes()));
                layers.get(i).nodes.get(j).addErrors(numNodes);
                layers.get(i).nodes.get(j).addInput(numNodes);
                if (j != numNodes - 1)
                {
                    layers.get(i).getNode(j).randomizeWeights(data.numAttributes());
                } else
                {
                    layers.get(i).getNode(j).randomizeWeights(4);  // output node has 4 weights           
                }
            }
        }

        // This set of doubles is for the layer input values
        set = new double[data.numAttributes()];

        //#####################
        // This outer loop is here to iterate learning over and over!!! ###
        // ####################
        for (int z = 0; z < 1; z++)//try 10,000 // Remember to update the testing part
        {
            // Obvious
            data.randomize(new Random(1));

            //Finds the division numbers for 70-30
            total = data.numInstances();
            seventy = total * .7;
            theRest = total - (int) seventy;

            //Assigns the training and test datasets
            trainingData = new Instances(data, 0, (int) seventy);
            testingData = new Instances(data, (int) seventy, theRest);

            /**
             * ******************************************************************
             * This loop is for going through each instance in the training data
             * ***********************************************************************
             */
            for (int k = 0; k < 3; k++)
            {
                System.out.println("\n#### New INSTANCE ###### " + k);
                //Just getting the values for the first input value
                for (int i = 0; i < trainingData.numAttributes()-1; i++)
                {
                    System.out.println(trainingData.instance(k).value(i) + " i " + i + " <-- " + trainingData.numAttributes() + " $ " + trainingData.instance(k).value(i) + " - trainingData.instance(k).value(i)");
                    set[i] = trainingData.instance(k).value(i);
                    layers.get(0).nodes.get(i).setInput(trainingData.instance(k).value(i), i);
                }

                // This right answer will change whenever the next instance is loaded for training
                rightAnswer = set[set.length - 1];

                for (int i = 0; i < numLayers; i++)
                {
                    for (int j = 0; j < layers.get(i).numWeights; j++)
                    {
                        layers.get(i).nodes.get(i).setInput(trainingData.instance(k).value(i), i);

                    }

                    //This does the layer feed forward part
                    for (int j = 0; j < layers.get(i).nodes.size(); j++)
                    {
                        System.out.println(i + " i and j " + j + "  nodes size: " 
                                + layers.get(i).getNode(i).inputValues);
                        
                        layers.get(i).nodes.get(j).calculateValues(layers.get(i).getNode(j).inputValues);
                    
                        System.out.println("values " + layers.get(i).getNode(j).values
                                + " ##weights " + layers.get(i).getNode(j).weights);
                        System.out.println(set[j] + "<--set[j] b4  node output : "
                                + layers.get(i).getNode(j).nodeOutput);
                        
                        // Set the new input values for the next instance

                        set[j] = layers.get(i).getNode(j).nodeOutput;
                    }
                }

                // This is the guessing part
                double highest = -1.0;
                double guess = 0;
                // layer.get( ) is still hardcoded for 2 layers
                for (int i = 0; i < layers.get(numLayers - 1).nodes.size(); i++)
                {
                    if (layers.get(1).getNode(i).nodeOutput > highest)
                    {
                        highest = layers.get(1).getNode(i).nodeOutput;
                        guess = i;
                    }
                }

                System.out.println(z + " Guess: " + guess + "...Answer: " + rightAnswer);
                /**
                 * ************************************************************
                 * Back-propagation part
                 * *************************************************************
                 */
                System.out.println("\n\nBack-propagation part");
                double error;

                // Layer level, starting with output layer
                for (int i = numLayers - 1; i >= 0; i--)
                {
                    // Which node in the layer
                    for (int j = 0; j < layers.get(i).nodes.size(); j++)
                    {
                        // Determines output node or not
                        if (i == (numLayers - 1))
                        {
                            // This is the error function
                            double a = layers.get(i).nodes.get(j).nodeOutput;
                            double t;

                            // Setting t, or the target output for the node
                            if (j == (int) rightAnswer)
                            {
                                System.out.println("oright " + j + ' ' + rightAnswer);
                                t = 1;
                            } else
                            {
                                System.out.println("owrong " + j + ' ' + rightAnswer);
                                t = 0;
                            }

                            // error for the whole node
                            error = a * (1 - a) * (a - t);

                            // Number of errors needed to be the same as the weights minus the bias node
                            for (int m = 0; m < layers.get(i).nodes.get(j).weights.size() - 2; m++)
                            {
                                layers.get(i).nodes.get(j).setError(m, error);
                            }

                            // reset each weight in the node
                            for (int m = 0; m < layers.get(i).nodes.get(j).weights.size() - 1; m++)
                            {
                                double newWeight;
                                // the .2 is the learning curve thing
                                newWeight = layers.get(i).nodes.get(j).weights.get(m) - (.2 * error * a);
                                layers.get(i).nodes.get(j).setWeight(m, newWeight);
                                System.out.println("a " + layers.get(i).nodes.get(j).weights.get(m));
                            }
                            System.out.println("------------------------------");
                        } else
                        {
                            // This is the error function
                            double a = layers.get(i).nodes.get(j).nodeOutput;

                            // set each weight in the node
                            for (int m = 0; m < layers.get(i).nodes.get(j).weights.size() - 1; m++)
                            {
                                // Set the error
                                error = a * (1 - a) * (layers.get(i).nodes.get(j).prevWeights.get(m) * 1);
                                layers.get(i).nodes.get(j).setError(m, error);

                                double newWeight;
                                // the .1 is the learning curve thing
                                newWeight = layers.get(i).nodes.get(j).weights.get(m) - (.2 * error * a);
                                layers.get(i).nodes.get(j).setWeight(m, newWeight);
                                System.out.println("b " + layers.get(i).nodes.get(j).weights.get(m));
                            }
                            System.out.println("------------------------------");
                        }
                    }
                }
            }// end of iteration

            /**
             * ****************************
             * This will do the testing part, and calculate an accuracy
             * ******************
             */
            if (z > 999995) // Remember to update the training part
            {
                System.out.println("in testing........");
                countRight = 0;
                for (int k = 0; k < testingData.numInstances(); k++)
                {
                    for (int i = 0; i < testingData.numAttributes(); i++)
                    {
                        System.out.println(testingData.instance(k).value(i) + " - trainingData.instance(k).value(i)");
                        set[i] = (double) testingData.instance(k).value(i);
                    }

                    // This right answer will change whenever the next instance is loaded for training
                    rightAnswer = set[set.length - 1];

                    for (int i = 0; i < numLayers; i++)
                    {
                        // layers.get(i).setLayerNumber(i);
                        //This does the layer feed forward part
                        for (int j = 0; j < layers.get(i).nodes.size(); j++)
                        {
                            layers.get(i).nodes.get(j).calculateValues(layers.get(i).getNode(j).inputValues);
                            // Set the new input values for the next instance
                            set[j] = layers.get(i).getNode(j).nodeOutput;
                        }
                    }

                    // This is the guessing part
                    double highest = -1.0;
                    double guess = 0;
                    // layer.get( ) is still hardcoded for 2 layers
                    for (int i = 0; i < layers.get(numLayers - 1).nodes.size(); i++)
                    {
                        if (layers.get(1).getNode(i).nodeOutput > highest)
                        {
                            highest = layers.get(1).getNode(i).nodeOutput;
                            guess = i;
                        }
                    }
                    if (guess == rightAnswer)
                    {
                        System.out.println("correct");
                        countRight++;
                    } else
                    {
                        System.out.println("wrong");
                    }
                    System.out.println(z + " Guess: " + guess + "...Answer: " + rightAnswer);
                }
                System.out.println("Accuracy: " + (double) countRight / (double) theRest);
            }
        } // end of training
    }

    /*
     ***********************************************************************************
     *****************************************************************************
     **********************************NODE***************************************
     *****************************************************************************
     ***********************************************************************************
     */
    public static class Node
    {
        List<Double> inputValues;
        List<Double> weights;
        List<Double> prevWeights;
        List<Double> errors;
        List<Double> values;
        double nodeOutput;

        public Node(int numAttributes)
        {
            weights = Collections.synchronizedList(new ArrayList());
            prevWeights = Collections.synchronizedList(new ArrayList());
            errors = Collections.synchronizedList(new ArrayList());
            values = Collections.synchronizedList(new ArrayList());
            inputValues = Collections.synchronizedList(new ArrayList());
            nodeOutput = -1.0;

            for (int i = 0; i < numAttributes; i++)
            {
                values.add(-1.0);
            }
        }

        void randomizeWeights(int numWeights)
        {
            Random random = new Random();
            for (int i = 0; i < numWeights; i++)
            {
                Double weight;
                weight = -1 + (1 - (-1)) * random.nextDouble();
                weights.add(weight);

                // The prev weights need initial values too
                prevWeights.add(0.0);
            }
        }

        void calculateValues(List<Double> input)
        {
            System.out.println("calculate values^#^#^#^ " + input);
            for (int i = 0; i < input.size() - 1; i++)
            {
                System.out.println(i + "values " + input);

                Double value;
                value = input.get(i) * (double) weights.get(i);
                values.set(i, value);
            }

            // This is adding the bias node to the end
            values.set(input.size() - 1, (-1.0 * weights.get(weights.size() - 1)));

            double temp = 0;
            for (int i = 0; i < input.size(); i++)
            {
                temp += values.get(i);
            }
            nodeOutput = 1 / (1 + Math.pow(Math.E, (-temp)));
            System.out.println("node opput&%@!# "+ nodeOutput);
        }

        void setWeight(int num, double newWeight)
        {
            prevWeights.set(num, weights.get(num));
            weights.set(num, newWeight);
            System.out.println("prev Weight " + prevWeights.get(num) + "---- New Weight " + newWeight);
        }

        // Set the new error
        void setError(int num, double error)
        {
            errors.set(num, error);
        }

        //This pretty much sets the placeholders for the errors
        void addErrors(int num)
        {
            for (int i = 0; i < num; i++)
            {
                errors.add(0.0);
            }
        }

        // Set the input
        void setInput(double value, int position)
        {
            System.out.println("val, pos " + value + " " + position);
            inputValues.add(position, value);
            //inputValues.
            System.out.println("inputvalues: " + inputValues);
        }

        // Placeholders for the input values
        void addInput(int num)
        {
            for (int i = 0; i < num; i++)
            {
                inputValues.add(0.0);
            }
        }
    }

    /**
     ***********************************************************************
     *****************************LAYER***********************************
     * **********************************************************************
     */
    public static class Layer
    {
        List<Node> nodes;
        Double bias = -1.0;
        int layerNumber;
        private int numWeights;

        public Layer()
        {
            nodes = Collections.synchronizedList(new ArrayList());
        }

        void addNode(Node n)
        {
            if (n != null)
            {
                boolean whatever;
                whatever = nodes.add(n);
            }
        }

        int getNumNodes()
        {
            return nodes.size();
        }

        // get a specific node in the node list
        Node getNode(int i)
        {
            if (i <= nodes.size())
            {
                return nodes.get(i);
            } else
            {
                Node nn = new Node(0);
                return nn;
            }
        }

        void setNumWeights(Instance inst)
        {
            numWeights = inst.numAttributes();
        }

        void setLayerNumber(int num)
        {
            layerNumber = num;
        }

        /**
         * @return the numWeights
         */
        int getLayerNumber()
        {
            return layerNumber;
        }

        /**
         * @return the numWeights
         */
        public int getNumWeights()
        {
            return numWeights;
        }
    }
}
