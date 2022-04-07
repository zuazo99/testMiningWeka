package Sailkatzailea;

import weka.Run;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Random;


public class ParametroEkorketa {

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Programaren helburua:");
            System.out.println("Erabilera:");
            System.out.println("java -jar ParametroEkorketa.jar train.arff emaitzak.txt ");


        } else {


            long startTime = System.nanoTime();

            // 1. Entrenamendurako datuak kargatu
            DataSource dataSource = new DataSource(args[0]);
            Instances data = dataSource.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);


            // 2. Klase minoritarioa lortu
            int min = Integer.MAX_VALUE;
            int minClassIndex = 0;
            for (int i = 0; i < data.numClasses(); i++) {
                int x = data.attributeStats(data.classIndex()).nominalCounts[i];
                System.out.println(data.attribute(data.classIndex()).value(i) + "-->"
                        + x + " instantzia kopurua" );

                if(x < min){
                    min = x;
                    minClassIndex = i;
                }
            }

            System.out.println("Klase minoritario: " + data.attribute(data.classIndex()).value(minClassIndex));
            double max = 0.0;
            double maxTime = Double.MAX_VALUE;

            FileWriter file = new FileWriter(args[1]);
            PrintWriter pw = new PrintWriter(file);

            pw.println();
            pw.println("RandomForest Parametro Ekorketa:");
            pw.println("Ekortuko ditugun parametroak:");
            pw.println("bagSizePercent maxDepth numFeatures numIterations FMEASURE DENBORA");
            pw.println("Ebaluzio metrika: Klase minoritarioa fMeasure");
            pw.println();
            System.out.println("bagSizePercent maxDepth numFeatures numIterations \t\tFMEASURE \t DENBORA(s)");


            RandomForest randomF = new RandomForest();
            randomF.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());
            int maxBSPB = 0;
            int maxMD = 0;
            int maxNF = 0;
            int maxNI = 0;
            for (int bspb = 1; bspb < 20; bspb += 5) { //10etik gora badoa exekuzio denbora asko handitzen da
                randomF.setBagSizePercent(bspb);
                for (int md = 10; md <= 60; md += 10) { //beste probetan ikusi dugu limitea 60 baino gehiago izanik puntuazioa ez dela aldatzen, beraz, ez da inoiz 100-era iristen
                    randomF.setMaxDepth(md);
                    for (int nf = 0; nf < data.numAttributes() - 1; nf +=50) { //-1 agian klasea kontuan har dezakelako
                        randomF.setNumFeatures(nf);
                        for (int ni = 1; ni < 50; ni += 5) {
                            long konbinazioHasieraDenbora = System.nanoTime();
                            randomF.setNumIterations(ni);
                //-----------------------HOLD-OUT--------------------------------------------//

                            // Randomize aplikatu
                            Randomize filterRandomize = new Randomize();
                            filterRandomize.setInputFormat(data);
                            Instances randomData = Filter.useFilter(data, filterRandomize);
                            // Train multzoak lortu
                            RemovePercentage filterTrain = new RemovePercentage();
                            filterTrain.setInvertSelection(false);
                            filterTrain.setPercentage(30);
                            filterTrain.setInputFormat(data);
                            Instances train = Filter.useFilter(randomData, filterTrain);

                            //Test multzoa lortu
                            RemovePercentage filterTest = new RemovePercentage();
                            filterTest.setInvertSelection(true);
                            filterTest.setPercentage(30);
                            filterTest.setInputFormat(data);
                            Instances test = Filter.useFilter(randomData, filterTest);
                            test.setClassIndex(test.numAttributes() - 1);
                            // Sailkatzailea entrenatu
                                randomF.buildClassifier(train);

                            //Ebaluazioa egin
                            Evaluation evaluator = new Evaluation(train);
                            evaluator.evaluateModel(randomF, test);
                            System.out.println(evaluator.toSummaryString("\n=== Results ===\n",false));

                            long konbinazioAmaieraDenbora = System.nanoTime();
                            double denbora = ((double) konbinazioAmaieraDenbora - konbinazioHasieraDenbora) / 1000000000;
                            System.out.println();
                            System.out.format("%10s \t %10s \t%10s \t%10s %20s \t%10s", bspb, md, nf, ni, evaluator.fMeasure(minClassIndex), denbora);
                            if (evaluator.fMeasure(minClassIndex) > max) { //si la puntuacion es mejor, pasa a ser el mejor

                                max = evaluator.fMeasure(minClassIndex);
                                maxTime = denbora;
                                maxBSPB = bspb;
                                maxMD = md;
                                maxNF = nf;
                                maxNI = ni;

                            } else if (evaluator.fMeasure(minClassIndex) == max && denbora < maxTime) { //si la puntuacion es igual, y el tiempo menor, pasa a ser el mejor
                                max = evaluator.fMeasure(minClassIndex);
                                maxTime = denbora;
                                maxBSPB = bspb;
                                maxMD = md;
                                maxNF = nf;
                                maxNI = ni;
                            }
                            pw.println();
                            pw.println("\t" + bspb + "\t \t" + md + "\t \t " + nf + "\t  " + ni + "\t   " + evaluator.fMeasure(minClassIndex) + "\t " + denbora);

                        }

                    }
                }
            }
            long stopTime = System.nanoTime();
            pw.println("\n Balio hoberenak: \n");
            pw.println("BSPB = " + maxBSPB + " MD = " + maxMD + " NF = " + maxNF + " NI = " + maxNI + " hurrengo puntuazioarekin " + max + " eta " + maxTime + " segundu behar izan ditu\n");
            pw.println("Exekuzio denbora: " + ((double) stopTime - startTime) / 1000000000 + " segundu");
            System.out.println("Exekuzio denbora: " + ((double) stopTime - startTime) / 1000000000);
            pw.flush();
            pw.close();


        }
    }


}