import data_structure
from data_helpers import parserNcbiTxtFile_simple,load_dict

from options import opt
import codecs
if __name__ == "__main__":

    # load cdr corpus and dictionary

    # testdocuments = load_txtdocuments(opt.test_file)
    # load_nlpdocuments(opt.testnlp_file, testdocuments, True)

    testdocuments = parserNcbiTxtFile_simple(opt.test_file)
    dict = load_dict(opt.dict_file)

    resultpath = "./output/20190106/2_85.71_nodigit_piehaoqianjiakongge"
    # resultpath = "/home/lyx/py-workspace/softmax_norm_ncbi_1029/output/1101/2_85.13_pubmed_sieve_word2digit"
    # resultpath = "/home/lyx/py-workspace/softmax_norm_ncbi_1029/output/1101/1_7718_pipelin_sieve_word2digit1021_all"
    # resultpath = "/home/lyx/py-workspace/softmax_norm_ncbi/output/1019/gold_norm_16"

    resultEntities = []
    correctEntities = []
    wrongEntities = []
    for line in codecs.open(resultpath, 'r', 'utf-8'):
        linesplit = line.split("\t")
        lineEntity = data_structure.Entity()
        lineEntity.doc_id = linesplit[0]
        lineEntity.start = linesplit[1]
        lineEntity.end = linesplit[2]
        lineEntity.text = linesplit[3]
        lineEntity.pre_meshId = linesplit[5].strip()
        resultEntities.append(lineEntity)
    totalEntityNum = 0
    goldEntities = []
    for document in testdocuments:
        totalEntityNum += len(document.entities)
        for entity in document.entities:
            goldEntities.append(entity)

    for preEntity in resultEntities:

        isfind = False
        for document in testdocuments:
            if preEntity.doc_id == document.doc_name:
                for entity in document.entities:

                    if preEntity.start == str(entity.start) and preEntity.end == str(entity.end):
                        isfind = True
                        preEntity.gold_meshId = entity.gold_meshId
                        if preEntity.pre_meshId == entity.gold_meshId:
                            correctEntities.append(preEntity)
                        else:
                            wrongEntities.append(preEntity)
                        break


    print("totalEntityNum:%s, preEntityNum:%s, correctNum:%s ,wrongNum: %s"%(totalEntityNum, len(resultEntities),  len(correctEntities), len(wrongEntities)))
    correctPath = "./output/error_analysis/correct_85.17"
    wrongpath = "./output/error_analysis/wrong_85.17"
    goldpath = "./output/error_analysis/goldentity"
    prepath = "./output/error_analysis/preentity_85.17"

    with open(correctPath, 'w') as f1:
        for entity in correctEntities:
            line = entity.doc_id + "\t" + entity.start + "\t" + entity.end + "\t"+ entity.text + "\t meshId" + entity.pre_meshId + "\t gold_Id " + entity.gold_meshId+"\n"

            f1.write(line)

    with open(wrongpath, 'w') as f2:
        for entity in wrongEntities:
            # line = entity.doc_name + "\t" + entity.start + "\t" + entity.end + "\t" + entity.text + "\t meshId " + entity.pre_meshId + "\t gold_Id " + entity.gold_meshId + "\n"
            line = entity.doc_id + "\t" + entity.start + "\t" + entity.end + "\t" + entity.text + "\n"

            pre_names = dict.id_to_names[entity.pre_meshId]
            pre_names_line = "|".join([pre_name for pre_name in pre_names])

            gold_names = dict.id_to_names[entity.gold_meshId]
            gold_names_line = "|".join([gold_name for gold_name in gold_names])


            f2.write(line)
            f2.write("pre_meshId " +" "+ entity.pre_meshId + " "  + pre_names_line + "\n")
            f2.write("gold_meshId"+" "+ entity.gold_meshId+ " " + gold_names_line+"\n")

    with open(goldpath, 'w') as f3:

        for entity in goldEntities:
            line = entity.doc_id + "\t" + str(entity.start)  + "\t" + str(entity.end) + "\t" + entity.text +"\t" + "meshId " + entity.gold_meshId + "\n"
            f3.write(line)
    # with open(prepath, 'w') as f4:
    #
    #     for entity in resultEntities:
    #         line = entity.doc_name + "\t" + entity.text + "\t meshId " + entity.pre_meshId + "\n"
    #         f4.write(line)