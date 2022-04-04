package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


public class getARFF {
    public static void main(String[] args)throws Exception {

        //args[0] --> train.csv
        //args[1] --> train.arff


        /*
         * takes 2 arguments:
         *  CSV input file
         *  ARFF output file
         */

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
                System.out.println("\n\nFiltered data:\n\n" + data);


                // save ARFF
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);
                saver.setFile(new File(args[1]));
                saver.writeBatch();


    }




    // Karaktere itsusiak garbitzeko metodoa.

    private static void removeCharactersFromFile(String fileName, String fileResult) throws IOException{
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        PrintWriter pw = new PrintWriter(fileResult);
        String line;

        while ((line = br.readLine()) != null) {
            // line = line.replace(subString, "");
            line = line.replaceAll("[`'?]", "");
            pw.println(line);
        }
        br.close();
        pw.close();
    }
}
