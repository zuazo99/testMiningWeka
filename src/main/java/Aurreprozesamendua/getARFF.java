package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;


public class getARFF {

    /**
     * 2 Parametro behar ditu programak:
     *
     * 	1. csv-a dagoen path-a
     * 	2. sortu nahi dugun train.arff-ren path-a
     *
     * @param args the arguments
     * @throws IOException Signals that an I/O exception has occurred.
     * @throws Exception Signals that an exception has occurred
     */

    public static void main(String[] args)throws Exception {

        //args[0] --> train.csv
        //args[1] --> train.arff

        // ** -- CSV garbiketa -- ** //

        if(args.length !=2 ) {
            System.out.println("Gordinik datorren train CSV emanda , train.arff\n"
                    + "lortu behar da");
            System.out.println("Aurrebaldintzak:");
            System.out.println("	*Parametro bat sartzea");
            System.out.println("\t1- Lehenengo parametro bezala existitzen den .csv fitxategiaren helbidea pasatzea.");
            System.out.println("\t2- Bigarren parametro bezala .arff fitxategia gorde nahi den helbidea existitzea.");
            System.out.println("\t4- Datu sortaren atributuen ordena: (1)identifikatzailea, (2) Textua, (3) Klasea");
            System.out.println("Postbaldintzak:");
            System.out.println("*Arff bat bueltan");
            System.out.println("Argumentuak: CSV fitxategia bat sartzea, arff-a gordeko den path-a");
            System.out.println("\nErabilera adibidea komando-lerroan:");
            System.out.println("java -jar GetARFF.jar train.csv train.arff");
        }



//            String path = "./Test";
//            File pathAsFile = new File(path);
//            if (!Files.exists(Paths.get(path)))
//                pathAsFile.mkdir();


            String path = args[0].substring(0, args[0].length() - 4);
            path = path+ "_Egokituta.csv";
            System.out.println(path);

            if (args[0].contains("test")){
                removeCharactersFromFileTest(args[0], path);
            }
        // 1. Fitxategia moldatu: Karaktere bereziak edo arazoak eman ditzaketenak kendu.

            removeCharactersFromFile(args[0], path);

            //  2. Load CSV fitxategia

                CSVLoader loader = new CSVLoader();

            try {
                loader.setSource(new File(path)); // CSV fitxategia kargatu
            }catch (IOException e){
                System.out.println(" Errorea: Sarrerako .csv fitxategiaren helbidea ez da zuzena");;
            }
                Instances data = loader.getDataSet();


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

                data.setClassIndex(data.numAttributes() - 1);
                System.out.println("\n\nFiltered data:\n\n" + data);

            // 4. fitxategia ARFF formatuan gorde

                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);
                saver.setFile(new File(args[1]));
                saver.writeBatch();


    }

    /**
     * Testua Arff formatura pasa.
     *
     * @param fileName daukazun csv fitxategiaren izena
     * @param fileResult sortuko den fitxategiaren izena
     * @throws IOException Signals that an I/O exception has occurred.
     */

    private static void removeCharactersFromFileTest(String fileName, String fileResult) throws IOException{
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        PrintWriter pw = new PrintWriter(fileResult);
        String line;

        while ((line = br.readLine()) != null) {
            line = line.replaceAll("[`'?.]", "").replaceAll("Nan", "");
            pw.println(line);
        }
        br.close();
        pw.close();
    }

    // Karaktere itsusiak garbitzeko metodoa.

    private static void removeCharactersFromFile(String fileName, String fileResult) throws IOException{
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
}
