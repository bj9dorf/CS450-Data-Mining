/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentalshellhardcodedclassifier;

import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

/**
 *
 * @author Bruce
 */
public class ExperimentalShellHardCodedClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String filepath = null;
        Instances data = null;

        DataSource source;
        try {
            source = new DataSource("C:\\Users\\Bruce\\Documents\\School Spring 2015\\CS450 Mining\\iris.csv");
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
        data.randomize(new Random(1));

        //Finds the division numbers for 70-30
        int total = data.numInstances();
        double seventy = total * .7;
        int theRest = total - (int) seventy;

        //Assigns the training and test datasets
        Instances dataSeventy = new Instances(data, 0, (int) seventy);
        Instances dataTheRest = new Instances(data, (int) seventy, theRest);

        HCClassifier hcc = new HCClassifier();
        try {
            hcc.buildClassifier(data);
        } catch (Exception e) {
            System.out.println("Exception thrown: " + e);
        }
        try {
            hcc.classifyInstance(data);
        } catch (Exception e) {
            System.out.println("Exception thrown: " + e);
        }

//        for (int i = 0; i < data.numInstances(); i++) {
//            double pred = 0;
//            try{
//                pred = hcc.classifyInstance(data.instance(i));
//            }
//            catch (Exception e)
//            {
//                System.out.println("Exception caught: " + e);
//            }
//            System.out.print("ID: " + data.instance(i).value(0));
//            System.out.print(", actual: " + data.classAttribute().value((int) data.instance(i).classValue()));
//            System.out.println(", predicted: " + data.classAttribute().value((int) pred));
//        }
        System.out.println("\nDataset:\n");
        System.out.println(dataSeventy + " : " + dataTheRest);
        System.out.println(dataSeventy.numInstances() + " : " + dataTheRest.numInstances());
    }
}
