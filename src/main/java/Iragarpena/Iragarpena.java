package Iragarpena;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

import java.io.*;

public class Iragarpena {

    /**
     * 4 Parametro behar ditu programak:
     *
     * 	1. modeloa
     * 	2. csv fitxategia
     * 	3. irteera fitxategia
     * 	4. hiztegia
     *
     * @param args the arguments
     * @throws Exception Signals that an exception has occurred
     */

    public static void main (String[] args) throws Exception {

         /*
            arg 1 modeloa
            arg 2 .csv fitxategia edo esaldi bat
            arg 3 = irteera fitxategia
            arg 4 = hiztegia

            ./modeloa/modeloaRandomForest.model ./datuak/test.csv ./modeloa/iragarpen.txt ./Dictionary/hiztegia.txt
         */

        if (args.length !=0){
            RandomForest randomF = (RandomForest) weka.core.SerializationHelper.read(args[0]);
            File file;
            FileWriter fw = new FileWriter(new File(args[2]));
            Instances data;
            Instances dataClear;
            if(args[1].contains(".csv")){
                String path = args[1].substring(0, args[1].length() - 4);
                path = path+ "_Egokituta.csv";
                System.out.println(path);

                removeCharactersFromFile(args[1], path);

               // data = datuakKargatu(args[1]); // .arff sartu
                data = getCSVLoader(path);
                data.setClassIndex(data.numAttributes() - 1);
                dataClear = data;

                // 3. Atributuak egokitu

                // 3.1 Lehenengo atributua, 'id', ez da beharrezkoa, ondorioz ezabatu egingo da Remove filtroa erabilita.

                Remove remove = new Remove();
                remove.setAttributeIndices("1"); //1.posizioko atributua ezabatu nahi da.
                remove.setInputFormat(data);
                remove.setInvertSelection(false); // Zehaztutako atributua nahi da, eta besteak mantendu.
                data = Filter.useFilter(data, remove);

                /*
                 3.2. 'text' atributua Nominal bezala kargatzen denez, String motara bihurtu behar dugu horretarako
                 NominalToString filtroa erabilita.
                 NominalToString
                 Atributu nominala String-era pasatu
                 */

                NominalToString filterToString = new NominalToString();
                filterToString.setInputFormat(data);
                filterToString.setOptions(Utils.splitOptions("-C 1")); // 1. posizion dago 'text' atributua
                data = Filter.useFilter(data, filterToString);

                NumericToNominal filterToNominal = new NumericToNominal();
                filterToNominal.setInputFormat(data);
                filterToNominal.setOptions(Utils.splitOptions("-R 2"));
                data = Filter.useFilter(data, filterToNominal);

                data = addValues(data);

                System.out.println("Filtered data: " + data);


                FixedDictionaryStringToWordVector filtroa = new FixedDictionaryStringToWordVector();
                filtroa.setDictionaryFile(new File(args[3]));
                filtroa.setInputFormat(data);
                data = Filter.useFilter(data, filtroa);

                Reorder reorder = new Reorder();
                reorder.setAttributeIndices("2-last,1");
                reorder.setInputFormat(data);
                data = Filter.useFilter(data, reorder);

                data.setClassIndex(data.numAttributes() - 1);
               // System.out.println("Filtered data: " + data);

                // 2. Eredua kargatu: getClassifier

                Classifier model = getClassifier(args[0]);

                //3. Ebaluatu eta iragarpenak egin
                Evaluation eval = ebaluatu(randomF, data);

                //4. Iragarpenak fitxategian idatzi
                fw = new FileWriter(args[2]);
                fw.write("Exekuzio data: "+java.time.LocalDateTime.now().toString()+"\n");
                fw.write("\n-- Test Set -- \n");
                fw.write("Instantzia\tActual\tPredicted\n");

                int i = 0;
                for (Prediction p: eval.predictions() ){
                    System.out.println(dataClear.instance(i).attribute(1).value((int) dataClear.instance(i).value(1))+ " " + data.attribute(data.classIndex()).value((int) p.predicted()));
                    fw.write(dataClear.instance(i).attribute(1).value((int) dataClear.instance(i).value(1))+ " " + data.attribute(data.classIndex()).value((int) p.predicted())+"\n");
                    i++;
                }

                // 5. Informazio gehiago gorde
                fw.write("\n"+eval.toClassDetailsString()+"\n");
                fw.write("\n"+eval.toSummaryString()+"\n");
                fw.write("\n"+eval.toMatrixString()+"\n");

                fw.close();


            }else{ //esaldi bat kargatu

                ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("./Datuak/trec_clean.arff"); //hutsi dagoen .arff behar dugu
                data = dataSource.getDataSet();
                data.setClassIndex(data.numAttributes() - 1);
                System.out.println(data.numInstances());
                Instance galdera = new DenseInstance(data.numAttributes());
                galdera.setDataset(data);
                galdera.setValue(1, args[1]);
                galdera.setMissing(0);
                data.add(galdera); //esaldia duen instantzia sortu eta .arff-ra gehitu
                System.out.println("Instance: " + galdera);
                System.exit(0);

                dataClear=data;
                FixedDictionaryStringToWordVector filtroa = new FixedDictionaryStringToWordVector();
                filtroa.setDictionaryFile(new File(args[3])); //esto habria que cambiarlo por un argumento
                filtroa.setInputFormat(data);
                data= Filter.useFilter(data, filtroa);
                Evaluation eval = new Evaluation(data);
                eval.evaluateModel(randomF, data);
                int i = 0;
                for (Prediction p: eval.predictions() ){   //ez da beharrezkoa for loop hau, baina agian esaldi bat baino gehiagorekin funtzionatzeko inplementatuko dugu
                    System.out.println(dataClear.instance(i).attribute(1).value(0)+ " " + data.attribute(0).value((int) p.predicted()));
                    fw.write(dataClear.instance(i).attribute(1).value(0)+ ", " + data.attribute(0).value((int) p.predicted()));
                    i++;
                }
            }
            fw.close();

        }else {
            System.out.println("Ez duzu arguments atala behar bezala bete!");
            System.out.println("Erabilerak:");
            System.out.println("Lehenengoa: java -jar Predictions.jar modeloa.model test.arff emaitzak.txt hiztegia");
            System.out.println("Bigarrena: java -jar Predictions.jar modeloa.model \"Ebaluatu nahi den esaldia\" emaitzak.txt hiztegia");

        }


    }

    /**
     * iragarpenak lortzeko.
     *
     * @param s daukazun csv fitxategiaren izena
     * @throws Exception Signals that an exception has occurred
     */


    public String iragarpenakAtera(String s) throws Exception {
        String iragarpena = "SPAM";
        RandomForest randomF = (RandomForest) weka.core.SerializationHelper.read("randomForestUI.model");
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("spam_clean.arff"); //hutsi dagoen .arff behar dugu
        Instances data = dataSource.getDataSet();
        data.setClassIndex(0);
        System.out.println(data.numInstances());
        Instance algo = new DenseInstance(data.numAttributes());
        algo.setDataset(data);
        algo.setValue(1, s);
        algo.setMissing(0);
        data.add(algo); //esaldia duen instantzia sortu eta .arff-ra gehitu
        Instances dataClear=data;
        FixedDictionaryStringToWordVector filtroa = new FixedDictionaryStringToWordVector();
        filtroa.setDictionaryFile(new File("hiztegiaUI.txt")); //esto habria que cambiarlo por un argumento
        filtroa.setInputFormat(data);
        data= Filter.useFilter(data, filtroa);
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(randomF, data);
        for(Prediction p: eval.predictions() ){
            if(p.predicted() == 1.0){
                iragarpena = "Ez SPAM";
            }
        }
        return iragarpena;
    }

    /**
     * Testua datuak kargatzeko.
     *
     * @param path datuen path-a
     * @throws Exception Signals that an exception has occurred
     */

    public static Instances datuakKargatu(String path) throws Exception{
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * ebaluatzeko
     *
     * @param model modeloa
     * @param test testa
     * @throws Exception Signals that an exception has occurred
     */

    public static Evaluation ebaluatu(Classifier model, Instances test) throws Exception{
        // Ebaluatu
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(model, test);
        return eval;
    }

    /**
     * classifier lortzeko
     *
     * @param path datuen path-a
     * @throws Exception Signals that an exception has occurred
     */

    public static Classifier getClassifier(String path) throws Exception{
       Classifier cls =  (Classifier) weka.core.SerializationHelper.read(path);
       return cls;
    }

    /**
     * characters removal
     *
     * @param fileName input datuen izena
     * @param fileResult output datuen izena
     * @throws Exception Signals that an exception has occurred
     */

    private static void removeCharactersFromFile(String fileName, String fileResult) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        PrintWriter pw = new PrintWriter(fileResult);
        String line;

        while ((line = br.readLine()) != null) {
            // line = line.replace(subString, "");
            line = line.replaceAll("[`'?.]", "");
            pw.println(line);
        }
        br.close();
        pw.close();
    }

    /**
     * CSV kargatzeko
     *
     * @param path datuen path-a
     * @throws Exception Signals that an exception has occurred
     */

    private static Instances getCSVLoader(String path) throws IOException {
        CSVLoader csvLoader = new CSVLoader();

        try {
            csvLoader.setSource(new File(path));
        }catch (IOException e){
            System.out.println(" Errorea: Sarrerako .csv fitxategiaren helbidea ez da zuzena");
        }
        Instances data = csvLoader.getDataSet();
        return data;
    }

    private static Instances addValues(Instances data) throws Exception{
        AddValues filterAdd = new AddValues();
        filterAdd.setOptions(Utils.splitOptions("-L DESC,ENTY,ABBR,HUM,NUM,LOC"));
        filterAdd.setInputFormat(data);
        data = Filter.useFilter(data, filterAdd);
        return data;
    }
}