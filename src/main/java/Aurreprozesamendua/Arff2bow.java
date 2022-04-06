package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


public class Arff2bow {

    public static void main(String[] args) throws Exception {

        /*
            args[0] --> train.arff
            args[1] --> hiztegia.txt
            args[2] --> TrainBOW.arff
            args[3] --> devRAW.arff
         */

        if (args.length !=5) {
            System.out.println("Programaren helburua: ");
            System.out.println("\tParametro bezala pasatzen diogun .arff fitxategiaren atributu espazioa errepresentazioa bektorial batera aldatu (BOW edo TF.IDF)");
            System.out.println("Aurrebaldintzak:");
            System.out.println("\t1- Lehenengo parametro bezala .arff fitxategia");
            System.out.println("\t2- Zein errepresentazioa bektoriala nahi den BOW edo TF.IDF");
            System.out.println("\t3- Sortutako hiztegia.txt non gordeko den path-a");
            System.out.println("\t4- Laugarren parametro bezala Sparse edo Non-Sparse emaitza fitxategi bezala nahi dugun.");
            System.out.println("\t5- Sortutako .arff fitxategiaren patha-a");
            System.out.println("\nErabilera adibidea komando lerroa-n");
            System.out.println("\tjava -jar TextToVector.jar <train.arff> 1 <outputPath hiztegia.txt> yes <outputPath BOW.arff>");

        }else{

            String arffFile = args[0];
            String bektorea = args[1];
            String dictionary = args[2];
            String bowArff = args[4];
            String devRAWFile = "./Datuak/devRAW.arff";

            // 1. Datuak kargatu(dataRAW)
            DataSource source = new DataSource(arffFile);
            Instances dataRaw = source.getDataSet();

            // Atributuaren izena aldatu behar dugu, StringToWordVector egiterako orduan arazoak ez izateko.
            dataRaw.renameAttribute(dataRaw.numAttributes() - 1, "etiqueta");
            dataRaw.setClassIndex(dataRaw.numAttributes() - 1);


            // Hemen holOut egiten dugu trainRAW eta devRAW lortzeko.
            ArrayList<Instances> datuakRAW = holdOut(dataRaw);
            Instances trainRAW = datuakRAW.get(0);
            Instances devRAW = datuakRAW.get(1);
            trainRAW.setClassIndex(trainRAW.numAttributes() - 1);


            System.out.println("Atributu kopurua: "+ trainRAW.numAttributes());
            System.out.println("Num instantziak trainRAW: "+ trainRAW.numInstances());
            System.out.println("Num instantziak devRAW: "+ devRAW.numInstances());

            File hiztegia = new File(dictionary);

            // ** StopWords ** //
            // https://weka.sourceforge.io/doc.dev/weka/core/Stopwords.html

        /*
            String2Word vector filtroa sortu
         */
            StringToWordVector filter = new StringToWordVector(); // RAW-tik bektore formatura
            filter.setInputFormat(trainRAW);
            filter.setLowerCaseTokens(true);
            filter.setTFTransform(false);
            filter.setIDFTransform(false);
            filter.setDictionaryFileToSaveTo(hiztegia);
            Instances trainBOW = Filter.useFilter(trainRAW, filter);
            System.out.println("\n\nFiltered data:\n\n" + trainBOW);

            //TrainRAW.arff save

            //TrainBOW.arff save
            datuakGorde(bowArff, trainBOW);
            // devRAW.arff save
            datuakGorde(devRAWFile, devRAW);
        }


    }

    private static void datuakGorde(String path, Instances data) throws IOException {
        ArffSaver s = new ArffSaver();
        s.setInstances(data);
        s.setFile(new File(path));
        s.writeBatch();
    }


    private static ArrayList<Instances> holdOut(Instances data) throws Exception{

        Randomize filterRandom = new Randomize();
        filterRandom.setRandomSeed(1); //esto se usa para el for aqui se coloca la i sino hay for se pone 1
        filterRandom.setInputFormat(data); //siempre que modifiques algo le recuerdas al filtro su formato
        Instances RandomData = Filter.useFilter(data, filterRandom);

        RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(RandomData); //Preparas el filtro.
        filterRemove.setPercentage(30); //Ajustas la cantidad de datos que quieres borrar --> En este caso --> 30% borras y te quedas 70%
        Instances train = Filter.useFilter(RandomData,filterRemove);


        filterRemove.setInputFormat(RandomData);
        filterRemove.setPercentage(30);
        filterRemove.setInvertSelection(true);
        Instances test = Filter.useFilter(RandomData,filterRemove);


        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1 );


        return new ArrayList<Instances>(){{
            add(train);
            add(test);
        }};
    }
}
