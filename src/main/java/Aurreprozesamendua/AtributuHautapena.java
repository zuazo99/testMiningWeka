package Aurreprozesamendua;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class AtributuHautapena {

    public static void main(String[] args) throws Exception {

        System.out.println(args.length);
        if(args.length  !=3) {
            System.out.println("Programaren helburua:");
            System.out.println("\tEntrenamendu multzoko atributu egokienak hautatu eta dev/test datasete-ra egokitu");
            System.out.println("\nAurrebaldintzak:");
            System.out.println("\t1- Lehenengo parametro bezala train.arff fitxategia");
            System.out.println("\t2- Bigarren parametro bezala sortuko den hiztegi.txt moldatuta atributu berrietara fitxategia.");
            System.out.println("\t3- Hirugarren parametro bezala test/dev.arff fitxategia.");
            System.out.println("\nPost baldintzak:");
            System.out.println("\t1- Parametroa BoW motako train arff fitxategia.");
            System.out.println("\t2- Parametroa Hiztegia gordetzeko fitxategia.");
            System.out.println("\t3- Dev/Test.arff fitxategia");
            System.out.println("\nErabilera adibidea komando-lerroan:");
            System.out.println("java -jar AtributuHautapena.jar train.arff hiztegia dev/test.arff");

            /*
                1. Parametroa BoW motako train arff fitxategia
                2. Parametroa Hiztegia gordetzeko fitxategia
                3. Dev/Test .arff fitxategia
             */
        }
        else {
            System.out.println(args[0]);

            String pathTRAIN = args[0].substring(0, args[0].length() - 5);
            pathTRAIN = pathTRAIN+ "_AtributuHautapena.arff";
            System.out.println(pathTRAIN);

            String pathDEV = args[2].substring(0, args[2].length() - 5);
            pathDEV = pathDEV+ "_AtributuHautapena.arff";
            System.out.println(pathDEV);

            DataSource dataSource = new DataSource(args[0]);
            Instances trainFSS = dataSource.getDataSet();
            trainFSS.setClassIndex(trainFSS.numAttributes() - 1);

            AttributeSelection attSelect = new AttributeSelection();
            InfoGainAttributeEval infoGainEval = new InfoGainAttributeEval();
            Ranker search = new Ranker();
            search.setOptions(new String[]{"-T", "0.001"});

            attSelect.setInputFormat(trainFSS);
            attSelect.setEvaluator(infoGainEval);
            attSelect.setSearch(search);
            trainFSS = Filter.useFilter(trainFSS, attSelect);

            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(trainFSS);
            arffSaver.setFile(new File(pathTRAIN));
            arffSaver.writeBatch();

            System.out.println(trainFSS.instance(0));
            // Hiztegi berria
            BufferedWriter bw = new BufferedWriter(new FileWriter(args[1]));

            for (int i = 0; i < trainFSS.numAttributes() - 1; i++) {
                Attribute a = trainFSS.attribute(i);
                bw.newLine();
                bw.write(a.name());
            }
            bw.flush();
            bw.close();



            DataSource devSource = new DataSource(args[2]);
            Instances dev = devSource.getDataSet();
            dev.setClassIndex(dev.numAttributes() - 1);


            FixedDictionaryStringToWordVector hiztegia = new FixedDictionaryStringToWordVector();
            hiztegia.setDictionaryFile(new File(args[1]));
            hiztegia.setInputFormat(dev);
            dev = Filter.useFilter(dev, hiztegia);

            RemoveWithValues filterRemoveValues = new RemoveWithValues();
            filterRemoveValues.setInputFormat(trainFSS);
            dev = Filter.useFilter(dev, filterRemoveValues);


            for (int i = 0; i < devSource.getDataSet().numInstances(); i++) {
                dev.add(i, devSource.getDataSet().instance(i));
            }

            arffSaver = new ArffSaver();
            arffSaver.setInstances(dev);
            //arffSaver.setDestination(new File("AtributuHautapena_" + args[2].toString()));
            arffSaver.setFile(new File(pathDEV));
            arffSaver.writeBatch();
        }
    }




}