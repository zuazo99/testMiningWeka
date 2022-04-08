package Sailkatzailea;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;


public class GetJ48Model {

    public static void main(String[] args) throws Exception {

        if(args.length != 4) {
            System.out.println("Programaren helburua:");
            System.out.println("\tBaseline sortu J48 erabiliz, lor daitekeen kalitatearen behe bornea ezartzeko");
            System.out.println("\nAurrebaldintzak:");
            System.out.println("\t1- Lehenengo parametro bezala train.arff fitxategia");
            System.out.println("\t2- Bigarren parametro bezala test.arff fitxategia");
            System.out.println("\t3- Bigarren parametro bezala eredu iragarlearen .model fitxategia gordetzeko path-a.");
            System.out.println("\t4- Hirugarren parametro bezala kalitatearen estimazioa gordetzeko .txt fitxategiaren path-a");
            System.out.println("\nPost Baldintzak:");
            System.out.println("\t1- Bigarren parametroan adierazitako path-an sortutako .model fitxategia gordeko da.");
            System.out.println("\t2- Hirugarren parametroan adierazitako path-an sortutako .txt fitxategia gordeko da.");
            System.out.println("\nArgumentuen zerrenda eta deskribapena:");
            System.out.println("\t1- Sarrerako train.arff fitxategiaren helbidea");
            System.out.println("\t2- Irteerako eredu iragarlearen .model fitxategiaren helbidea");
            System.out.println("\t3- Irteerako .txt fitxategiaren helbidea");
            System.out.println("\nErabilera:");
            System.out.println("java -jar GetJ48Model.jar train.arff test.arff modeloa.model kalitatearenestimazioa.txt ");
        }
        else{
            DataSource source=null;
            try {
                source = new DataSource(args[0]);
            } catch (Exception e) {
                System.out.println("train multzoa sortzeko sartu duzun arff-aren helbidea okerra da.");
            }

            Instances train = source.getDataSet();
            train.setClassIndex(train.numAttributes()-1);

            DataSource source1 = new DataSource(args[1]);

            Instances test = source1.getDataSet();

            test.setClassIndex(test.numAttributes() - 1);
            //##########################################################

            J48 model = new J48();
            model.buildClassifier(train);
            //##########################################################

            weka.core.SerializationHelper.write(args[2], model);


            FileWriter fw = new FileWriter(args[3]);


            //1- Ebaluazioa normala
            Evaluation evalTrainDev = new Evaluation(train);
            evalTrainDev.evaluateModel(model, train);
            fw.write("\n=============================================================\n");
            fw.write("EBALUAZIO EZ ZINTZOA:\n");
            fw.write(evalTrainDev.toSummaryString()+"\n");
            fw.write(evalTrainDev.toClassDetailsString()+"\n");
            fw.write(evalTrainDev.toMatrixString()+"\n");

            //2- Cross Validation
            Evaluation evaluatorCross = new Evaluation(train);

            //##########################################################
            model = new J48();
            model.buildClassifier(train);
            //##########################################################

            evaluatorCross.crossValidateModel(model, train, 2, new Random(1));

            fw.write("\n=============================================================\n");
            fw.write("CROSS VALIDATION-EKIN EBALUATUZ (TRAIN MULTZOAN SOILIK):\n");
            fw.write(evaluatorCross.toSummaryString()+"\n");
            fw.write(evaluatorCross.toClassDetailsString()+"\n");
            fw.write(evaluatorCross.toMatrixString()+"\n");


            //3-HOLD OUT





                Evaluation evaluatorSplit = new Evaluation(train);
            for(int i = 0; i<10; i++){
                Randomize filter = new Randomize();
                filter.setRandomSeed(0);
                filter.setInputFormat(train);
                train = Filter.useFilter(train, filter);

                RemovePercentage rmpct = new RemovePercentage();
                rmpct.setInputFormat(train);
                rmpct.setInvertSelection(false);
                rmpct.setPercentage(30);
                Instances train1 = Filter.useFilter(train, rmpct);

                //test multzoa
                RemovePercentage rmpct2 = new RemovePercentage();
                rmpct2.setInputFormat(train);
                rmpct2.setInvertSelection(true);
                rmpct2.setPercentage(30);
                Instances test1 = Filter.useFilter(train, rmpct2);

                //##########################################################
                model = new J48();
                model.buildClassifier(train1);
                //##########################################################
                //evaluation
                evaluatorSplit.evaluateModel(model, test1);
            }



            //Fitxategian gorde kalitatearen estimazioa
            fw.write("\n=============================================================\n");
            fw.write("HOLD OUT-EKIN (%70) EBALUATUZ:\n");
            fw.write(evaluatorSplit.toSummaryString()+"\n");
            fw.write(evaluatorSplit.toClassDetailsString()+"\n");
            fw.write(evaluatorSplit.toMatrixString()+"\n");
            fw.flush();
            fw.close();
        }

    }

}