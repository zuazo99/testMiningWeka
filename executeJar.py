import subprocess

subprocess.call(['java', '-jar', './jar/testMiningWeka.jar', './modeloa/modeloaRandomForest.model', './datuak/test.csv', './modeloa/iragarpen.txt'
                 , './Dictionary/hiztegia.txt'])