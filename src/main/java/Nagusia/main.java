package Nagusia;

import Aurreprozesamendua.Arff2bow;
import Aurreprozesamendua.AtributuHautapena;
import Aurreprozesamendua.MakeCompatible;
import Aurreprozesamendua.getARFF;
import Sailkatzailea.GetJ48Model;
import Sailkatzailea.GetRandomForestModel;
import Sailkatzailea.ParametroEkorketa;
import Iragarpena.Iragarpena;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;


public class main {

    private static Scanner sc = new Scanner(System.in);
    private static String workspace = System.getProperty("user.home")+"/IdeaProjects/testMiningWeka";


    public static void main (String args[]) throws Exception{


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
            System.out.println("Hurrengo direktorioa erabiliko da: "+ workspace);
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
        System.out.println("4. Parametro ekorketa egin RandomForest modeloa eraikitzeko");
        System.out.println("5. J48 modeloa lortzeko");
        System.out.println("6. RandomForest modeloa lortzeko");
        System.out.println("7. Iragarpenak egin");
        System.out.println("8. Exit");
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

    private static void aukera1() throws Exception {
        String [] parametroak = {workspace+"/Datuak/train.csv",workspace+"/Datuak/train.arff"};
        getARFF.main(parametroak);
        System.out.println("train.arff fitxategia sortu egin da");
    }

    private static void aukera2() throws Exception {

        String [] par1 = null;

        par1 = new String[]{"./Datuak/train.arff","BOW","./Dictionary/hiztegia.txt","Sparse","./Datuak/trainBOW.arff"};
        Arff2bow.main(par1);

        String [] par2 = {"./Datuak/devRAW.arff","./Datuak/trainBOW.arff","./Dictionary/hiztegia.txt","./Datuak/devBOW.arff"};
        MakeCompatible.main(par2);


        System.out.println("Arff gordina errepresentazio bektorialera pasatzea lortu egin da");
    }

    private static void aukera3() throws Exception {
        String [] parametroak = {"./Datuak/trainBOW.arff","./Dictionary/hiztegiaFSS.txt","./Datuak/devBOW.arff"};
        AtributuHautapena.main(parametroak);
        System.out.println("Atributu hautapena egin da");
    }

    private static void aukera4() throws Exception {
        String [] parametroak = {"./Datuak/trainBOW.arff","./modeloa/emaitzakParamEkorketa.txt"};
        ParametroEkorketa.main(parametroak);
        System.out.println("ParametroEkorketa egin da");
    }

    private static void aukera5() throws Exception {
        String [] parametroak = {"./Datuak/trainBOW.arff","./Datuak/test.arff","./modeloa/modeloaJ48.model","./modeloa/emaitzakJ48.txt"};
        GetJ48Model.main(parametroak);
        System.out.println("Baseline sortu J48 erabiliz, lor daitekeen kalitatearen behe bornea");
    }

    private static void aukera6() throws Exception {
        String [] parametroak = {"./Datuak/trainBOW.arff","./modeloa/modeloaRandomForest.model","./modeloa/emaitzakRForest.txt"};
        GetRandomForestModel.main(parametroak);
        System.out.println("Baseline sortu Random Forest erabiliz, lor daitekeen kalitatearen behe bornea");
    }

    private static void aukera7() throws Exception {
        String [] parametroak = {"./modeloa/modeloaRandomForest.model","./Datuak/test.csv","./modeloa/iragarpen.txt","./Dictionary/hiztegia.txt"};
        Iragarpena.main(parametroak);
        System.out.println("Iragarpena ondo egin da.");
    }

    private static void aukera8() throws Exception {
        System.exit(0);
    }
}
