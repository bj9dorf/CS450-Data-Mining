/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentalshell;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Bruce
 */
public class KNNClassifier extends Classifier {

    private Instances trainingData;
    private Instances testingData;
    private int k; //number of nearest neigbors to find

    private void setTrainingData(Instances data) {
        trainingData = data;
    }

    private void setTestingData(Instances data) {
        testingData = data;
    }

    public Instances getTrainingData() {
        return trainingData;
    }

    public Instances getTestingData() {
        return testingData;
    }

    public void setKNN(int num) {
        k = num;
    }

    public int getKNN() {
        return k;
    }

    @Override
    public void buildClassifier(Instances data) throws java.lang.Exception {
        data.randomize(new Random(1));

        //Finds the division numbers for 70-30
        int total = data.numInstances();
        double seventy = total * .7;
        int theRest = total - (int) seventy;

        //Assigns the training and test datasets
        setTrainingData(new Instances(data, 0, (int) seventy));
        setTestingData(new Instances(data, (int) seventy, theRest));
    }

    @Override
    public double classifyInstance(Instance data)
            throws java.lang.Exception {
        double type = 1.0;
//        Instances nnSet = null;// I couldn't use this...
        Instance temp = null;
        List distances = Collections.synchronizedList(new ArrayList());
        List items = Collections.synchronizedList(new ArrayList());
        int lastAttribute = data.numAttributes() - 1;
        double tempDist;
        
        for (int i = 0; i < trainingData.numInstances() - 1; i++) {
            tempDist = 0;
            //    d= (w1-w2)^2 + (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
            for (int j = 0; j < trainingData.numAttributes() - 1; j++) {
                tempDist = tempDist + Math.pow((data.value(j) - trainingData.instance(i).value(j)), 2);
                temp = trainingData.instance(i);
            }
            if (i >= k) {
                for (int m = 0; m < distances.size(); m++) {
                    if ((double) distances.get(m) > tempDist) {
                        //System.out.println(i+"::" + distances.get(m) + ": has been removed. " + tempDist + " : has been added.");
                        distances.remove(m);
                        distances.add(tempDist);
                        items.remove(m);
                        items.add(temp);
                    }
                }
            } else {
                //System.out.println(tempDist + " has automatically been added");
                distances.add(tempDist);
                items.add(temp);
            }

        }
        if (k == 1) {
            type = ((Instance)items.get(0)).value(lastAttribute);
       }
//        else
            
        return type;
    }
}
