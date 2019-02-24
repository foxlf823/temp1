import codecs
from data_structure import *
from utils import clean_str,wordToDigit,get_sentences_and_tokens_from_nltk
from options import opt
import logging
import nltk

def load_dict(path):
    dict = Dictionary()
    concepts =[]
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue

        concept = Concept()

        linesplit = line.split("\t")
        concept.meshId = linesplit[0].strip()

        name_synonym = list()
        for idx in range(1, len(linesplit)):
            names = linesplit[idx]

            names = clean_str(names)
            if opt.use_word2digit:
                names = wordToDigit(names)
            if names == '':
                continue
            name_synonym.append(names)
        concept.set_names(name_synonym)
        concepts.append(concept)

    dict.set_concepts(concepts)
    dict.set_id_to_names()
    return dict

def loadAbbreviations(abbrePath):
    abbreviations = list()
    lines = codecs.open(abbrePath, 'r', 'utf-8')
    for line in lines:
        line = line.strip().lower()
        if line=='':
            continue
        linesplits = line.split("\t")
        abbre = DiseaseAbbreviation()

        if len(linesplits) < 3:
            print(line)
        linesplits[1] = clean_str(linesplits[1])
        linesplits[2] = clean_str(linesplits[2])
        if opt.use_word2digit:
            linesplits[1] = wordToDigit(linesplits[1])
            linesplits[2] = wordToDigit(linesplits[2])

        abbre.initAbbre(linesplits[0].strip(), linesplits[1], linesplits[2])
        if abbre not in abbreviations:
            abbreviations.append(abbre)
    return abbreviations

def preprocessMentions(traindocuments, devdocuments, testdocuments, abbreviations):
    # abbreviation replace
    for doc in traindocuments:
        for entity in doc.entities:
            for abbre in abbreviations:
                if doc.doc_name == abbre.docId:
                    if entity.text == abbre.sf:
                        entity.text = abbre.lf
                        break

    for doc in devdocuments:
        for entity in doc.entities:
            for abbre in abbreviations:
                if doc.doc_name == abbre.docId:
                    if entity.text == abbre.sf:
                        entity.text = abbre.lf
                        break

    for doc in testdocuments:
        for entity in doc.entities:
            for abbre in abbreviations:
                if doc.doc_name == abbre.docId:
                    if entity.text == abbre.sf:
                        entity.text = abbre.lf
                        break

def parserNcbiTxtFile_simple(path):
    logging.info("loadData: {}".format(path))
    if opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')

    documents = []
    id=title=abstractt = ""
    document = Document()
    for line in codecs.open(path, 'r', 'utf-8'):

        line = line.strip()

        if line != "":
            linesplits = line.split("|")
            if len(linesplits) == 3:
                if linesplits[1] == "t":
                    id = linesplits[0]
                    title = linesplits[2]
                if linesplits[1] == "a":
                    abstractt = linesplits[2]
            linesplitsEntity = line.split("\t")
            if len(linesplitsEntity) == 6:
                meshId = linesplitsEntity[len(linesplitsEntity)-1]
                index = meshId.find(":")
                if index != -1:
                    meshId = meshId[index+1:]
                meshId = meshId.strip()
                entity = Entity()
                entitytext = clean_str(linesplitsEntity[3])
                if opt.use_word2digit:
                    entitytext = wordToDigit(entitytext)


                entity.setEntity(linesplitsEntity[0],int(linesplitsEntity[1]), int(linesplitsEntity[2]), entitytext,'Disease',meshId.strip())

                document.entities.append(entity)
        else:
            if len(id)>0 and len(title)>0 and len(abstractt)>0:
                document.initDocument(id, title, abstractt)
                if id == '2234245':
                    print(id)
                document_text = title + " " + abstractt
                sentences = get_sentences_and_tokens_from_nltk(document_text.lower(), nlp_tool, document.entities)
                document.sents = sentences
                document.initDocument(id, title, abstractt)

                documents.append(document)
                id = title = abstractt = ""
                document = Document()

    return documents

def readwrongresult(wrongfile):
    entities = []
    for line in codecs.open(wrongfile, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue
        linesplits = line.split("\t")
        if len(linesplits) == 6:
            entity = Entity()
            entity.doc_id = linesplits[0].strip()
            entity.start = int(linesplits[1])
            entity.end = int(linesplits[2])
            entity.text = linesplits[3].strip()
            entity.gold_meshId = linesplits[5].strip().split(" ")[1]
            entities.append(entity)


    return entities