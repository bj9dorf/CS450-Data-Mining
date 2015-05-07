/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentalshell;

//import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Bruce
 */
public class ExperimentalShell {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        String filepath = null;
        Instances data = null;
        KNNClassifier knn = new KNNClassifier();
        DataSource source;
        int lastAttribute;
        int correctGuesses = 0;
        double guess = 1.0;
        try {
            source = new ConverterUtils.DataSource("C:\\Users\\Bruce\\Documents\\School Spring 2015\\CS450 Mining\\iris.csv");
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
        
        int num = 1;
        //System.out.println("Please give the number of 'nearest neigbors': ");
        // num = whatever input;
        try {
            knn.buildClassifier(data);
            knn.setKNN(num);
        } catch (Exception e) {
            System.out.println("Exception thrown: " + e);
        }
        
        lastAttribute = data.numAttributes() - 1;
        try {
            for (int i = 0; i < knn.getTestingData().numInstances(); i++) {
                guess = knn.classifyInstance(knn.getTestingData().instance(i));
                if (knn.getTestingData().instance(i).value(lastAttribute) == guess) {
                    correctGuesses++;                    
                }
            }
        } catch (Exception e) {
            System.out.println("Exception thrown:: " + e);
        }
        System.out.println("Percent right: " + (1.0 * correctGuesses / knn.getTestingData().numInstances()));

//        for (int i = 0; i < data.numInstances() - 1; i++) {
//            double pred = 0;
//            try{
//                pred = knn.classifyInstance(data.instance(i));
//            }
//            catch (Exception e)
//            {
//                System.out.println("Exception caught: " + e);
//            }
//            System.out.print("ID: " + data.instance(i).value(0));
//            System.out.print(", actual: " + data.classAttribute().value((int) data.instance(i).classValue()));
//            System.out.println(", predicted: " + data.classAttribute().value((int) pred));
//        }
//        System.out.println("\nDataset:\n");
//        System.out.println(data);//Seventy + " : " + dataTheRest);
//        System.out.println(dataSeventy.numInstances() + " : " + dataTheRest.numInstances());
    }
}