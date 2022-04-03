package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;


public class TextToVector {

    public static void main(String[] args) throws Exception {

        /*
            args[0] --> train.arff
            args[1] --> hiztegia.txt
            args[2] --> TrainBOW.arff
         */

        DataSource source = new DataSource(args[0]);
        Instances dataRaw = source.getDataSet();
        System.out.println("\n\nImported data:\n\n" + dataRaw);
        ArrayList<Instances> datuakRAW = holdOut(dataRaw);
        System.out.println("Distantzia: "+ datuakRAW.size());

        Instances trainRAW = datuakRAW.get(0);
        Instances devRAW = datuakRAW.get(1);

        trainRAW.setClassIndex(trainRAW.numAttributes() - 1);
        System.out.println("Atributu kopurua: "+ trainRAW.numAttributes());
        System.out.println("Num instantziak trainRAW: "+ trainRAW.numInstances());
        System.out.println("Num instantziak devRAW: "+ devRAW.numInstances());

        // ** StopWords ** //
        // https://weka.sourceforge.io/doc.dev/weka/core/Stopwords.html

        File hiztegia = new File(args[1]);


        // String2Word vector filtroa sortu
        String[] options = new String[1];
        options[0] = "-R <1,9,10>";

        StringToWordVector filter = new StringToWordVector(); // RAW-tik bektore formatura
        System.out.println(filter.listOptions().asIterator().next().toString());

        filter.setInputFormat(trainRAW);
        filter.setLowerCaseTokens(true);
        filter.setDictionaryFileToSaveTo(hiztegia);
        Instances trainBOW = Filter.useFilter(trainRAW, filter);
        System.out.println("\n\nFiltered data:\n\n" + trainBOW);


        //TrainBOW.arff save
        ArffSaver s = new ArffSaver();
        s.setInstances(trainBOW);
        s.setFile(new File(args[2]));
        s.writeBatch();

//        FixedDictionaryStringToWordVector filterDictionary = new FixedDictionaryStringToWordVector();
//        filterDictionary.setOutputWordCounts(false);
//        filterDictionary.setLowerCaseTokens(true);
//        filterDictionary.setInputFormat(trainBOW);
//        filterDictionary.setDictionaryFile(hiztegia);
//
//        Instances devBOW = Filter.useFilter(devRAW, filterDictionary);
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
