package com.company;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class Main {

    int rangeStart = 1;
    int rangeEnd = 100;
    Class<?> cls = RandomForest.class;
    int multiplier = 1;
    String fileName = "data.arff";
    String optionName = "-I";
    public static void main(String[] args) throws Exception {
        Main main = new Main();
        main.tuneClassifier(main.rangeStart,
                main.rangeEnd,
                main.fileName,
                main.optionName,
                main.multiplier,
                main.cls);
    }

    private void tuneClassifier(int rangeStart, int rangeEnd, String fileName, String optionName, int multiplier, Class<?> cls) throws Exception {
        for (int i = rangeStart; i <= rangeEnd; i++) {
            BufferedReader reader = new BufferedReader(
                    new FileReader(fileName));
            Instances data = new Instances(reader);
            reader.close();
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
            String[] options = new String[2];
            options[0] = optionName;
            int multi = i* multiplier;
            options[1] = Integer.toString(multi);
            AbstractClassifier classifier = (AbstractClassifier) cls.newInstance();
            classifier.setOptions(options);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(1));
            System.out.println(multi + "\t" + eval.pctCorrect());
        }
    }
}
