import codecs
from options import opt
from data_helpers import parserNcbiTxtFile_simple

import data_structure
def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def generateEntities(sentence, label_matrix):
    docId = sentence.docId
    entities =[]

    for span in label_matrix:
        text=""
        start=-1
        end =-1
        span_start = span.index('[')+1
        span_end = span.index(']')
        span_str = span[span_start:span_end]
        if "," in span_str:
            start = int(span_str.split(",")[0].strip())
            end = int(span_str.split(",")[1].strip())
        else:
            start = int(span_str.strip())
            end = int(span_str.strip())

        entity_start = int(sentence.token_offset[start].split(" ")[0].strip())
        entity_end = int(sentence.token_offset[end].split(" ")[1].strip())

        for i in range(start, end+1):
            if i <end:
                text += sentence.tokens[i] + " "

            else:
                text += sentence.tokens[i]
        entity = data_structure.Entity()
        entity.create(docId,"Disease",entity_start, entity_end, text)
        entities.append(entity)
    return entities

def load_banner(banner_path, id_path):
    sentences = []
    doc_sentId_list = []
    sent_lines = []
    for line in codecs.open(banner_path, 'r', 'utf-8'):
        sent_lines.append(line.strip())

    ids = []
    for line in codecs.open(id_path, 'r', 'utf-8'):
        ids.append(line.strip())

    assert  len(sent_lines) == len(ids), 'not equal len sents and ids'
    for i, line in enumerate(sent_lines):
        sent = data_structure.Sent()
        docId_sentId = ids[i].split('-')
        assert len(docId_sentId)==2, 'len docid_sentid not 2'

        sent.docId = docId_sentId[0]
        sent.sentId = docId_sentId[1]
        if (docId_sentId[0],docId_sentId[1]) not in doc_sentId_list:
            doc_sentId_list.append((docId_sentId[0],docId_sentId[1]))
        if line=='':
            continue;
        tokens_labels = line.split()
        sent.tokens = [token_label.split('|')[0] for token_label in tokens_labels]
        for token_label in tokens_labels:
            if token_label.find('|')== -1:
                print(token_label)

        sent.labels = [token_label.split('|')[1] for token_label in tokens_labels]

        sent.text = ' '.join(token for token in sent.tokens)
        sentences.append(sent)

    assert len(doc_sentId_list) == len(sentences)
    doc_sentId_list_sort = sorted(doc_sentId_list)

    documents = []
    doc = data_structure.Document()
    doc_sents = []

    for doc_sent in doc_sentId_list_sort:
        assert len(doc_sent) == 2, 'docid, sentid not equal'
        docid, sentid = doc_sent
        if doc.doc_name != docid:
            if doc.doc_name == '':
                doc.doc_name = docid
            else:
                if len(doc_sents) != 0:
                    doc.sents = doc_sents
                    documents.append(doc)
                    doc = data_structure.Document()
                    doc_sents = []

        for temp_sent in sentences:
            if docid == temp_sent.docId and sentid == temp_sent.sentId:
                doc_sents.append(temp_sent)
                break

    if len(doc_sents) != 0:
        doc.sents = doc_sents
        documents.append(doc)


    #add each token start, end
    for i in range(len(documents)):
        doc_text = ' '.join(sent.text for sent in documents[i].sents)
        offset = 0
        for j in range(len(documents[i].sents)):
            if j != 0:
                offset += len(documents[i].sents[j-1].text) + 1
            documents[i].sents[j].start = offset
            buffer = ''
            token_offset = []
            for token in documents[i].sents[j].tokens:
                token_start = documents[i].sents[j].start + len(buffer)
                token_end = token_start + len(token)
                token_offset.append(str(token_start) + ' ' + str(token_end))

                buffer += token + ' '

            documents[i].sents[j].token_offset = token_offset

    return documents

def outputDocuments(outPath, documents):
    with open(outPath, 'w') as f:
        for doc in documents:
            f.write(doc.doc_name+"|t|" + doc.sents[0].text +'\n')
            abstractt = ' '.join([sent.text for i, sent in enumerate(doc.sents) if i>0])
            f.write(doc.doc_name+"|a|" + abstractt +'\n')
            for entity in doc.entities:
                f.write(doc.doc_name+"\t"+str(entity.start) +"\t"+str(entity.end)+
                        "\t"+entity.text+"\t"+entity.type +"\n")

def outputDocuments_title_abstract_entity(outPath, documents):
	with open(outPath, 'w',encoding='utf-8') as f:
		for doc in documents:
			if doc.title =='':
				print(doc.doc_name)
			f.write(doc.doc_name+"|t|" + doc.title +'\n')
			f.write(doc.doc_name+"|a|" + doc.abstractt + '\n')
			for entity in doc.entities:
				f.write(doc.doc_name+"\t"+str(entity.start) +"\t" + str(entity.end)+"\t"+entity.text+"\t"+'Disease' + "\t"+  entity.gold_meshId+ "\n")
			f.write('\n')

def outputDocuments_entities(outPath, documents,test_ids):
    test_ids.sort()
    with open(outPath, 'w') as f:
        for id in test_ids:
            for doc in documents:
                if id == doc.doc_name:
                    for entity in doc.entities:
                        f.write(doc.doc_name+"\t"+str(entity.start) +"\t"+str(entity.end)+
                                        "\t"+entity.text+"\t"+entity.type + "\n")

def outputDocuments_ner_entities(outPath, documents):
	with open(outPath, 'w') as f:
		for doc in documents:
			for entity in doc.entities:
				f.write(doc.doc_name+"\t"+str(entity.start) + "\t" +str(entity.end)+"\t"+entity.text+"\t"+'Disease'+"\t" +entity.gold_meshId+"\n")

def load_entity_doc(ner_path):
	entity_docs = []
	doc_id = ''
	entities = []
	doc = data_structure.Document()
	for line in codecs.open(ner_path, 'r', 'utf-8'):
		linesplits = line.strip().split('\t')
		if len(linesplits) == 0:
			continue

		if doc_id !=linesplits[0]:
			if doc_id == '':
				doc_id = linesplits[0]

			if len(entities) > 0:
				doc.doc_name = doc_id
				doc.entities = entities
				entity_docs.append(doc)
				entities = []
				doc = data_structure.Document()
				doc_id = linesplits[0]

		meshId = '-1'
		if len(linesplits) == 5:
			temp_idx = linesplits[4].find(':')
			if temp_idx != -1:
				meshId = linesplits[4][temp_idx + 1:]
			else:
				meshId = linesplits[4]


		entity = data_structure.Entity()
		entity.doc_name = linesplits[0]
		entity.start = int(linesplits[1])
		entity.end = int(linesplits[2])
		entity.text = linesplits[3]
		entity.gold_meshId = meshId
		entities.append(entity)

	if len(entities) > 0:
		doc.doc_name = doc_id
		doc.entities.extend(entities)
		entity_docs.append(doc)
	return entity_docs


if __name__ == '__main__':

	ner_path = "/home/lyx/workspace/Dnorm_ncbi/ncbi_test_plain_ner"
	output_path_doc = "./sample_data/ncbi_test_ner_evalNorm"
	output_path_entity = "/home/lyx/workspace/Dnorm_ncbi/ncbi_test_plain_ner_entities"

	ncbi_ner_path = "/home/lyx/workspace/Dnorm_ncbi/output/analysis_ncbi.txt"



	# entity_docs = load_entity_doc(ner_path)
	entity_docs = load_entity_doc(ncbi_ner_path)
	test_documents = parserNcbiTxtFile_simple(opt.test_file)


	for i in range(len(entity_docs)):
		isfind = False
		for test_doc in test_documents:
			if entity_docs[i].doc_name == test_doc.doc_name:
				isfind =True
				entity_docs[i].title = test_doc.title
				entity_docs[i].abstractt = test_doc.abstractt
				break
		if not isfind:
			print(entity_docs[i].doc_name)
	outputDocuments_title_abstract_entity(output_path_doc, entity_docs)
	# outputDocuments_ner_entities(output_path_entity, entity_docs)
	print('end')






    # banner_path = "/home/lyx/workspace/Dnorm_ncbi/output/cdr_training_0108.txt"
    # id_path = "/home/lyx/workspace/Dnorm_ncbi/output/cdr_ids_0108.txt"
    # out_path = "/home/lyx/py-workspace/softmax_norm_cdr_2019/sample_data/banner_cdr_test_0108"
    # out_path = "/home/lyx/py-workspace/softmax_norm_cdr_2019/sample_data/banner_cdr_test_0108_entities"

    # banner_path = "/home/lyx/workspace/Dnorm_ncbi/output/cdr_training_0109.txt"
    # id_path = "/home/lyx/workspace/Dnorm_ncbi/output/cdr_ids_0109.txt"
    # out_path = "/home/lyx/py-workspace/softmax_norm_cdr_2019/sample_data/banner_cdr_test_0109"

    # documents = load_banner(banner_path, id_path)
	#
    # docids =[]
    # for i in range(len(documents)):
    #     if documents[i].doc_name not in docids:
    #         docids.append(documents[i].doc_name)
    #     for sent in documents[i].sents:
    #         label_matrix = get_ner_BIO(sent.labels)
    #         entities = generateEntities(sent,label_matrix)
    #         documents[i].entities.extend(entities)
    # # outputDocuments(out_path, documents)
    # outputDocuments_entities(out_path, documents,docids)



