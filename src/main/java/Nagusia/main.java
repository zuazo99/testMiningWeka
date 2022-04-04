package Nagusia;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;

public class main {

    private static Scanner sc = new Scanner(System.in);
    private static String workspace = System.getProperty("user.home")+"/text_mining";


    public static void main (String args[]) throws Exception{
        //String workspace = System.getProperty("user.home")+"/text_mining";//añadir /?

        if (args.length != 1) {
            System.out.println("Defektuzko direktorioa erabiliko da (sortuko da ez existitzekotan): "+ workspace);
            System.out.println("Beste direktorioren bat nahiago izatekotan parametro gisa pasa diezaiokezu programa exekutatzerakoan: java -jar Main.jar lantokiarenHelbidea");
            System.out.println();
            pressEnterToContinue();
            if (Files.notExists(Paths.get(workspace)))
                karpetaSortu(workspace);
        }

        else {
            workspace = args[0];
            System.out.println("Hurrengo direktorioa erabiliko da: "+ workspace); //pedir /?
            pressEnterToContinue();
        }

        System.out.println("Ongi etorri!");
        System.out.println();

        String aukera;
        boolean bukatu=false;
        while (!bukatu) {
            menuaPantailaratu();
            aukera=sc.next();

            switch (aukera)
            {
                case "1": aukera1();
                    pressEnterToContinue();
                    break;
                case "2": aukera2();
                    pressEnterToContinue();
                    break;
                case "3": aukera3();
                    pressEnterToContinue();
                    break;
                case "4": aukera4();
                    pressEnterToContinue();
                    break;
                case "5": aukera5();
                    pressEnterToContinue();
                    break;
                case "6": aukera6();
                    pressEnterToContinue();
                    break;
                case "7": aukera7();
                    pressEnterToContinue();
                    break;
                case "8": aukera8();
                    pressEnterToContinue();
                    break;
                case "9": aukera9();
                    pressEnterToContinue();
                    break;
                case "10":  bukatu= true;
                    break;
                default: System.out.println("Sartu duzun aukera ez da existitzen");
                    pressEnterToContinue();
                    break;
            }
        }
    }

    private static void menuaPantailaratu() {
        System.out.println("\nAUKERAK:\n");
        System.out.println("1. Fitxategietatik arff gordina atera");
        System.out.println("2. Arff gordina errepresentazio bektorialera pasa");
        System.out.println("3. Atributu hautapena egin");
        System.out.println("4. Parametro ekorketa egin SMV modeloa eraikitzeko");
        System.out.println("5. Logistic Regression sailkatzailea entrenatu");
        System.out.println("6. Support Vector Machine(SVM) sailkatzailea entrenatu");
        System.out.println("7. Testa konpatibilizatu train formatuarekin");
        System.out.println("8. Iragarpenak egin");
        System.out.println("9. Visualizer");
        System.out.println("10. Exit");
    }


    private static void pressEnterToContinue()
    {
        System.out.println("Sakatu enter tekla jarraitzeko...");
        try
        {
            System.in.read();
        }
        catch(Exception e)
        {}
    }

    private static void karpetaSortu(String path) {
        File file = new File(path);

        boolean bool = file.mkdir();
        if(bool){
            System.out.println("Direktorioa ongi sortu da");
        }else{
            System.out.println("Ez da lortu direktorioa sortzea, barkatu eragozpenak");
        }
    }

    private static void aukera1() throws IOException, InterruptedException {
        String hola = "" ;
        System.out.println("\nSartu testu fitxategiak dauden direktorioaren path-a");
        String directorypath = sc.next();
        String [] parametroak = {directorypath,workspace};
        if (directorypath.contains(".txt")) {
            hola = "/testGordina.arff";
            parametroak[1]+= hola;
        }
        else {
            hola = "/trainGordina.arff";
            parametroak[1]+=hola;
        }

        GetRaw.main(parametroak);
        System.out.println(hola+" gorde da "+ workspace+"-n");
    }

    private static void aukera2() throws Exception {
        String [] parametroak = {workspace+"/trainGordina.arff",workspace+"/rawDictionary.txt","","",workspace+"/trainBektoreak.arff"};
        System.out.println("\nTFIDF erabili nahi? Bai/Ez (defektuz BoW)");
        if (sc.next().toLowerCase().equals("bai"))	parametroak[2] = "-I";
        System.out.println("\nNonSparse formatua erabili nahi? Bai/Ez (defektuz Sparse)");
        if (sc.next().toLowerCase().equals("bai"))	parametroak[3] = "-N";
        //for (String x : parametroak) System.out.println(x);	//parametroak ikusteko

        TransformRaw.main(parametroak);
        System.out.println("trainBektoreak.arff eta rawDictionary.txt gorde dira "+workspace+"-n");
    }

    private static void aukera3() throws Exception {
        System.out.println("\nLimite bat ezarri nahi? Bai/Ez (300 da maximoa)");
        if (sc.next().toLowerCase().equals("bai")) {
            System.out.println("Sartu zenbaki bat(negatiboak ez dira kontuan hartuko)");
            String [] parametroak = {workspace+"/trainBektoreak.arff",workspace+"/fss.arff" ,workspace+"/dictionary.txt",sc.next()};

            FSS.main(parametroak);
        }
        else {
            String [] parametroak = {workspace+"/trainBektoreak.arff",workspace+"/fss.arff" ,workspace+"/dictionary.txt"};

            FSS.main(parametroak);
        }
        System.out.println("dictionary.arff eta fss.arff gorde dira "+workspace+"-n");
    }

	/* private static void aukera3() throws Exception {
		 String [] parametroak = {workspace+"/trainBektoreak.arff",workspace+"/fss.arff" ,workspace+"/dictionary.txt","301"};
		 System.out.println("\nLimite bat ezarri nahi? Bai/Ez (300 da maximoa)");
		 if (sc.next().toLowerCase().equals("bai"))	parametroak[3] = sc.next();

		 FSS.main(parametroak);
	 }*/

    private static void aukera4() throws Exception {
        String [] parametroak = {workspace+"/fss.arff",workspace+"/parametroEkorketa.txt"};

        ParametroEkorketa.main(parametroak);
        System.out.println("parametroEkorketa.txt gorde da "+workspace+"-n");
    }

    private static void aukera5() throws Exception {
        String [] parametroak = {workspace+"/fss.arff",workspace+"/logisticRegression.model", workspace+"/TestPredictionsLogReg.txt"};

        LogisticRegression.main(parametroak);
        System.out.println("logisticRegression.model eta TestPredictionsLogReg.txt gorde dira "+workspace+"-n");
    }

    private static void aukera6() throws Exception {
        System.out.println("Gamma parametroa zehaztu (iradokizuna: 0.001)");
        String gamma = sc.next();
        System.out.println("C parametroa zehaztu (iradokizuka: 10)");
        String cost = sc.next();
        String [] parametroak = {workspace+"/fss.arff",workspace+"/SMO.model", workspace+"/TestPredictionsSVM.txt",gamma,cost};

        SMOModel.main(parametroak);
        System.out.println("SMO.model TestPredictionsSVM.txt gorde dira "+workspace+"-n");
    }

    private static void aukera7() throws Exception {
        String [] parametroak = {workspace+"/testGordina.arff",workspace+"/dictionary.txt", workspace+"/testBektoreak.arff"}; //makecompatible: tfidf esta, y nonsparse?

        MakeCompatible.main(parametroak);
        System.out.println("testBektoreak.arff gorde da "+workspace+"-n");
    }

    private static void aukera8() throws Exception {
        String [] parametroak = {workspace+"/testBektoreak.arff",workspace+"/SMO.model", workspace+"/predictions.txt"};

        SMOPredictions.main(parametroak);
        System.out.println("predictions.txt gorde da "+workspace+"-n");
    }

    private static void aukera9() throws Exception {
        System.out.println("Sartu arff baten path-a instantziak bistaratzeko");
        String [] parametroak = {sc.next()};
        VisualizeInstances.main(parametroak);
    }

}
}
