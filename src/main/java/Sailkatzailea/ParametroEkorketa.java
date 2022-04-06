package Sailkatzailea;

import weka.Run;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

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
            BufferedWriter bw = new BufferedWriter(new FileWriter(args[1]));
            bw.newLine();
            bw.write("bagSizePercent maxDepth numFeatures numIterations FMEASURE DENBORA");
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
                    for (int nf = 0; nf < data.numAttributes(); nf += 200) { //-1 agian klasea kontuan har dezakelako
                        randomF.setNumFeatures(nf);
                        for (int ni = 1; ni < 50; ni += 5) {
                            long konbinazioHasieraDenbora = System.nanoTime();
                            randomF.setNumIterations(ni);
                            Evaluation evaluator = new Evaluation(data);
                            evaluator.crossValidateModel(randomF, data, 10, new Random(1));
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
                            bw.newLine();
                            bw.write("\t" + bspb + "\t \t" + md + "\t \t " + nf + "\t  " + ni + "\t   " + evaluator.fMeasure(minClassIndex) + "\t " + denbora);

                        }

                    }
                }


            }
            long stopTime = System.nanoTime();
            bw.write("\n Balio hoberenak: \n");
            bw.write("BSPB = " + maxBSPB + " MD = " + maxMD + " NF = " + maxNF + " NI = " + maxNI + " hurrengo puntuazioarekin " + max + " eta " + maxTime + " segundu behar izan ditu\n");
            bw.write("Exekuzio denbora: " + ((double) stopTime - startTime) / 1000000000 + " segundu");
            System.out.println("Exekuzio denbora: " + ((double) stopTime - startTime) / 1000000000);
            bw.flush();
            bw.close();


        }
    }

}