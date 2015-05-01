/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentalshellhardcodedclassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
/**
 *
 * @author Bruce
 */
public class HCClassifier extends Classifier {
    @Override
    public void buildClassifier(Instances data) throws java.lang.Exception
    {
        
    }
    double classifyInstance(Instances data)
                        throws java.lang.Exception
    {
        return 1;
    }
}
