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
            System.out.println("Ez duzu arguments atala behar bezala bete!");
            System.out.println("Erabilera:");
            System.out.println("java -jar AtributuHautapena.jar train.arff hiztegia dev/test.arff ");            /*
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
            Instances train = dataSource.getDataSet();
            train.setClassIndex(train.numAttributes() - 1);

            AttributeSelection attSelect = new AttributeSelection();
            InfoGainAttributeEval infoGainEval = new InfoGainAttributeEval();
            Ranker search = new Ranker();
            search.setOptions(new String[]{"-T", "0.001"});

            attSelect.setInputFormat(train);
            attSelect.setEvaluator(infoGainEval);
            attSelect.setSearch(search);
            train = Filter.useFilter(train, attSelect);

            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(train);
            arffSaver.setFile(new File(pathTRAIN));
            arffSaver.writeBatch();

            System.out.println(train.instance(0));
            // Hiztegi berria
            BufferedWriter bw = new BufferedWriter(new FileWriter(args[1]));

            for (int i = 0; i < train.numAttributes() - 1; i++) {
                Attribute a = train.attribute(i);
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
            filterRemoveValues.setInputFormat(train);
            dev = Filter.useFilter(dev, filterRemoveValues);

//            Remove filterRemove = new Remove();
//            filterRemove.setInputFormat(train);
//            dev = Filter.useFilter(dev, filterRemove);

            for (int i = 0; i < train.numInstances(); i++) {
                dev.add(i, train.instance(i));
            }


            // Clasea azken atributu bezala ezarri
//            Reorder reorder = new Reorder();
//            reorder.setAttributeIndices("2-" + dev.numAttributes() + ",1");
//            reorder.setInputFormat(dev);
//            dev = Filter.useFilter(dev, reorder);


            arffSaver = new ArffSaver();
            arffSaver.setInstances(dev);
            //arffSaver.setDestination(new File("AtributuHautapena_" + args[2].toString()));
            arffSaver.setFile(new File(pathDEV));
            arffSaver.writeBatch();
        }
    }




}