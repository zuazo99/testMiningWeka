package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;


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

        if(args.length == 0){
            System.out.println("Gordinik datorren train CSV emanda , train.arff\n"
                    + "lortu behar da");
            System.out.println("Aurrebaldintzak:");
            System.out.println("	*Parametro bat sartzea");
            System.out.println("Postbaldintzak:");
            System.out.println("	*Arff bat bueltan");
            System.out.println("Argumentuak: CSV fitxategia bat sartzea, arff-a gordeko den path-a");
            System.out.println("java -jar GetARFF.jar train.csv train.arff");

        }else if (args.length == 2){
            removeCharactersFromFile(args[0], "./Datuak/trainResult.csv");

            //  load CSV
            try {

                CSVLoader loader = new CSVLoader();
                loader.setSource(new File("./Datuak/trainResult.csv"));
                //loader.setSource(new File(args[0]));
                Instances data = loader.getDataSet();



                // save ARFF
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);
                saver.setFile(new File(args[1]));
                saver.writeBatch();

            }catch (IOException e){
                e.printStackTrace();
            }
        }


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
