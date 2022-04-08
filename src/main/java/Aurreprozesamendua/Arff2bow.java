package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


public class Arff2bow {

    /**
     * 5 Parametro behar ditu programak:
     *
     * 	1. gure datuak train.arff dokumentuan gorde egiten dira
     * 	2. bektorea
     * 	3. atributu zenbakia pasatzeko
     * 	4. gure train arff-aren errepresentazio bektoriala
     * 	5. raw file-aren testa hold out-ekin sortua
     *
     * @param args the arguments
     * @throws Exception Signals that an exception has occurred
     */

    public static void main(String[] args) throws Exception {

        /*
            args[0] --> train.arff
            args[1] --> hiztegia.txt
            args[2] -->
            args[3] --> TrainBOW.arff
            args[4] --> devRAW.arff
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
            System.out.println("\tjava -jar TextToVector.jar <train.arff> BOW/IDF <outputPath hiztegia.txt> Sparse/NonSparse <outputPath BOW.arff> <devRAW.arff>");

        }else {
            String arffFile = args[0];
            String bektorea = args[1];
            String dictionary = args[2];
            String sparse = args[3];
            String bowArff = args[4];
            String devRAWFile = "./Datuak/devRAW.arff";
            String trainRAWFile = "./Datuak/trainRAW.arff";

            // 1. Datuak kargatu(dataRAW)
            DataSource source = new DataSource(arffFile);
            Instances dataRaw = source.getDataSet();

            // Atributuaren izena aldatu behar dugu, StringToWordVector egiterako orduan arazoak ez izateko.
            dataRaw.renameAttribute(dataRaw.numAttributes() - 1, "etiqueta");
            dataRaw.setClassIndex(dataRaw.numAttributes() - 1);


            // 2. Hemen holOut egiten dugu trainRAW eta devRAW lortzeko.
            ArrayList<Instances> datuakRAW = holdOut(dataRaw);
            Instances trainRAW = datuakRAW.get(0);
            Instances devRAW = datuakRAW.get(1);
            trainRAW.setClassIndex(0);


            System.out.println("Atributu kopurua: " + trainRAW.numAttributes());
            System.out.println("Num instantziak trainRAW: " + trainRAW.numInstances());
            System.out.println("Num instantziak devRAW: " + devRAW.numInstances());

            File hiztegia = new File(dictionary);

            // ** StopWords ** //
            // https://weka.sourceforge.io/doc.dev/weka/core/Stopwords.html

        /*
            3. BOW edo TF.IDF
            String2Word vector filtroa sortu
         */

            Instances trainBOW = null;

            if (bektorea.equals("BOW")) {
                // BOW
                trainBOW = stringtowordvector(trainRAW, false, hiztegia);
            } else if (bektorea.equals("IDF")) {
                // TF.IDF
                trainBOW = stringtowordvector(trainRAW, true, hiztegia);

            } else {
                System.out.println("Errorea: Bigarren parametroa ez da zuzena");
                System.exit(0);
            }

            /*
               4. Sparse edo NonSparse - 'Dispertsioa / Ez-Dispertsioa'
             */

            if (sparse.equals("Sparse")){ //Sparse
                // Defektuz horrela gordetzen da
                //TrainBOW.arff save
                datuakGorde(bowArff, trainBOW);

            }else if (sparse.equals("NonSparse")){ //NonSparse
                trainBOW = nonSparse(trainBOW);
                datuakGorde(bowArff, trainBOW);
            }else{
                System.out.println("Errorea: Laugarren parametroa ez da zuzena");
                System.exit(0);
            }
            System.out.println("\n\nFiltered data:\n\n" + trainBOW);

            //TrainRAW.arff save
            datuakGorde(trainRAWFile, trainRAW);
            // devRAW.arff save
            datuakGorde(devRAWFile, devRAW);

        }

    }

    /**
     * Datuak kargatzeko.
     *
     * @param path daukazun datuen path-a
     * @param data instantzien datuak
     * @throws Exception Signals that an exception has occurred
     */

    private static void datuakGorde(String path, Instances data) throws Exception {

        //REORDER atributuak

        Reorder reorder = new Reorder();
        reorder.setAttributeIndices("2-last,1");
        reorder.setInputFormat(data);
        data = Filter.useFilter(data, reorder);

        ArffSaver s = new ArffSaver();
        s.setInstances(data);
        s.setFile(new File(path));
        s.writeBatch();
    }


    /**
     * String-ak hitzetan bihurtzeko metodoa
     *
     * @param data Instantzien datuak
     * @param bool bool --> TRUE - TF.IDF kontuan hartu nahi dugu.
     * @param hiztegia atributu zenbakia pasatzeko
     * @throws Exception Signals that an exception has occurred
     */


    private static Instances stringtowordvector(Instances data, boolean bool, File hiztegia) throws Exception{
            StringToWordVector filter = new StringToWordVector(); // RAW-tik bektore formatura

            filter.setInputFormat(data);
            filter.setLowerCaseTokens(true); // Letra larria nahiz xehea baliokidetu
            filter.setIDFTransform(bool); //bool --> TRUE - TF.IDF kontuan hartu nahi dugu.
            filter.setDictionaryFileToSaveTo(hiztegia);
            Instances train = Filter.useFilter(data, filter);

            return train;
    }

    /**
     * Non sparse aplikatzen du
     *
     * @param data Instantzien datuak
     * @throws Exception Signals that an exception has occurred
     */

    private static Instances nonSparse(Instances data) throws Exception{
        // nonSparsera pasatzeko
        SparseToNonSparse filterNonSparse = new SparseToNonSparse();
        filterNonSparse.setInputFormat(data);
        Instances nonSparseData = Filter.useFilter(data,filterNonSparse);

        return nonSparseData;
    }

    /**
     * Hold out aplikatzen du
     *
     * @param data Instantzien datuak
     * @throws Exception Signals that an exception has occurred
     */

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
