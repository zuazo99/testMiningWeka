package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;
import java.io.IOException;

public class MakeCompatible {

    public static void main(String[] args) throws Exception {
        if(args.length  !=4) {
            System.out.println("Ez duzu arguments atala behar bezala bete!");
            System.out.println("Erabilera:");
            System.out.println("java -jar MakeCompatible.jar devRAW.arff trainBOW.arff hiztegia.txt devBOW.arff");
        } else{

            String devRAWfile = args[0];
            String trainBOWfile = args[1];
            String dictionary = args[2];
            String devBOWfile = args[3];

            // 1.Datuak kargatu: devRaw

            Instances devRAW = datuakKargatu(devRAWfile);
            Instances trainBOW = datuakKargatu(trainBOWfile);

            trainBOW.setClassIndex(trainBOW.numAttributes() - 1);
            devRAW.setClassIndex(devRAW.numAttributes() - 1);

            // 2. Atributuak reordenatu, klasea amaieran agertu dadin, horretarako reorder filtroa erabiliko dugu

//            Reorder  reorder = new Reorder();
//            reorder.setAttributeIndices("2-"+devRAW.numAttributes()+",1");
//            reorder.setInputFormat(devRAW);
//            devRAW = Filter.useFilter(devRAW, reorder);


            // 3. Hiztegia kargatu --> Dev-ean sartu egingo dugu FixedDictionaryStringToWordVector


            FixedDictionaryStringToWordVector filterDictionary = new FixedDictionaryStringToWordVector();
            filterDictionary.setDictionaryFile(new File(dictionary));
            filterDictionary.setOutputWordCounts(false);
            filterDictionary.setLowerCaseTokens(true);
            filterDictionary.setInputFormat(trainBOW);
           // filterDictionary.setOptions(Utils.splitOptions("-R first-last, -dictionary "+dictionary+", -L"));

            Instances devBOW = Filter.useFilter(devRAW, filterDictionary);
            datuakGorde(devBOWfile, devBOW);
        }

    }

    private static Instances datuakKargatu(String path) throws Exception{
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();
        return data;
    }

    private static void datuakGorde(String path, Instances data) throws IOException {
        ArffSaver s = new ArffSaver();
        s.setInstances(data);
        s.setFile(new File(path));
        s.writeBatch();
    }


}
