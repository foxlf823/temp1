
import cdr

from data_helpers import load_txtdocuments, load_nlpdocuments, loadAbbreviations, load_dict,parserNcbiTxtFile
from options import opt
import codecs
if __name__ == "__main__":
	trainDocuments = parserNcbiTxtFile(opt.train_file)
	devDocuments = parserNcbiTxtFile(opt.dev_file)
	testdocuments = parserNcbiTxtFile(opt.test_file)
	dict = load_dict(opt.dict_file)
	resultpath = "/home/lyx/py-workspace/softmax_norm_ncbi_1029/output/1101/2_85.13_pubmed_sieve_word2digit"
	resultEntities = []
	correctEntities = []
	wrongEntities = []
	for line in codecs.open(resultpath, 'r', 'utf-8'):
		linesplit = line.split("\t")
		lineEntity = cdr.Entity()
		lineEntity.doc_name = linesplit[0]
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

	wrongpath = "/home/lyx/py-workspace/softmax_norm_ncbi_1029/output/error_analysis/wrong_classify"

	with open(wrongpath, 'w') as f2:
		for i, entity in enumerate(wrongEntities):
			lineEntity = str(i) + ". " + entity.text + "\t pre_Id=" + entity.pre_meshId + "\t gold_Id=" + entity.gold_meshId + "\n"
			f2.write(lineEntity)
			linegoldnames = ""
			lineprenames = ""
			isgoldfind = False
			isprefind = False
			for (id, names) in dict.id_to_names.items():

				if entity.gold_meshId == id:

					isgoldfine = True
					for i, name in enumerate(names):
						if i<len(names)-1:
							linegoldnames += name+"|"
						else:
							linegoldnames += name

				if entity.pre_meshId == id:
					isprefind = True
					for i, name in enumerate(names):
						if i<len(names)-1:
							lineprenames += name+"|"
						else:
							lineprenames += name

				if isprefind and isgoldfind:
					break

			f2.write("\t" + entity.gold_meshId +" "+ linegoldnames + "\n")
			f2.write("\t" + entity.pre_meshId + " "+ lineprenames + "\n")
