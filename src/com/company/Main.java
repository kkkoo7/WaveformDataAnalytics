package com.company

import weka.classifiers.AbstractClassifier
import weka.classifiers.evaluation.Evaluation
import weka.core.Instances

import java.io.BufferedReader
import java.io.FileReader
import java.util.Random

class Main {

    internal var rangeStart = 1
    internal var rangeEnd = 100
    internal var cls: Class<*> = weka.classifiers.lazy.IBk::class.java
    internal var multiplier = 10
    internal var fileName = "data.arff"
    internal var optionName = "-K"

    @Throws(Exception::class)
    private fun tuneClassifier(rangeStart: Int, rangeEnd: Int, fileName: String, optionName: String, multiplier: Int, cls: Class<*>) {
        for (i in rangeStart..rangeEnd) {
            val reader = BufferedReader(
                    FileReader(fileName))
            val data = Instances(reader)
            reader.close()
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1)
            val options = arrayOfNulls<String>(2)
            options[0] = optionName
            val multi = i * multiplier
            options[1] = Integer.toString(multi)
            val classifier = cls.newInstance() as AbstractClassifier
            classifier.options = options
            val eval = Evaluation(data)
            eval.crossValidateModel(classifier, data, 10, Random(1))
            println(multi.toString() + "\t" + eval.pctCorrect())
        }
    }

    companion object {
        @Throws(Exception::class)
        @JvmStatic fun main(args: Array<String>) {
            val main = Main()
            //        main.tuneClassifier(main.rangeStart, main.rangeEnd, main.fileName, main.optionName, main.multiplier, main.cls);
            main.tuneClassifier(27,
                    40,
                    main.fileName,
                    "-K",
                    1,
                    weka.classifiers.trees.RandomForest::class.java)
        }
    }
}
