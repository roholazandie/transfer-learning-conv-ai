


# with open("/home/rohola/codes/transfer-learning-conv-ai/out/generated_sentences_logs_emotion_recog.tsv", 'w') as fw:
#     with open("/home/rohola/codes/transfer-learning-conv-ai/out/emotion_correlation_input.txt") as fr:
#         for line in fr:
#             predicted_sentence = line.split('\t')[0]
#             fw.write(predicted_sentence.rstrip()+"\t"+"fear"+"\n")


# with open("/home/rohola/codes/transfer-learning-conv-ai/out/true_sentences.tsv", 'w') as fw:
#     with open("/home/rohola/codes/transfer-learning-conv-ai/out/emotion_correlation_input.txt") as fr:
#         for line in fr:
#             true_sentence = line.split('\t')[1]
#             fw.write(true_sentence.rstrip()+"\t"+"fear"+"\n")


import spacy
nlp = spacy.load("en_core_web_sm")
# doc = nlp("Hello, world. Here are two sentences.")
# print(list(doc.sents)[0])

# with open("/home/rohola/codes/transfer-learning-conv-ai/out/first_sentence/true_sentences.tsv", 'w') as fw:
#     with open("/home/rohola/codes/transfer-learning-conv-ai/out/true_sentences.tsv") as fr:
#         for line in fr:
#             text = line.split('\t')[0].rstrip()
#             doc = nlp(text)
#             first_sentence = list(doc.sents)[0]
#             fw.write(first_sentence.text.rstrip() + "\tfear\n")

# with open("/home/rohola/codes/transfer-learning-conv-ai/out/first_sentence/generated_sentences_logs_em.tsv", 'w') as fw:
#     with open("/home/rohola/codes/transfer-learning-conv-ai/out/generated_sentences_logs_em.tsv") as fr:
#         for line in fr:
#             text = line.split('\t')[0].rstrip()
#             doc = nlp(text)
#             first_sentence = list(doc.sents)[0]
#             fw.write(first_sentence.text.rstrip() + "\tfear\n")

# with open("/home/rohola/codes/transfer-learning-conv-ai/out/first_sentence/generated_sentences_logs.tsv", 'w') as fw:
#     with open("/home/rohola/codes/transfer-learning-conv-ai/out/generated_sentences_logs.tsv") as fr:
#         for line in fr:
#             text = line.split('\t')[0].rstrip()
#             doc = nlp(text)
#             try:
#                 first_sentence = list(doc.sents)[0]
#                 fw.write(first_sentence.text.rstrip() + "\tfear\n")
#             except:
#                 first_sentence = "this is test."
#                 print("+")
#                 fw.write(first_sentence + "\tfear\n")


# with open("/home/rohola/codes/transfer-learning-conv-ai/out/first_sentence/generated_sentences_logs14.tsv", 'w') as fw:
#     with open("/home/rohola/codes/transfer-learning-conv-ai/out/generated_sentences_logs14.tsv") as fr:
#         for line in fr:
#             text = line.split('\t')[0].rstrip()
#             doc = nlp(text)
#             try:
#                 first_sentence = list(doc.sents)[0]
#                 fw.write(first_sentence.text.rstrip() + "\tfear\n")
#             except:
#                 first_sentence = "this is test."
#                 print("+")
#                 fw.write(first_sentence + "\tfear\n")


with open("/home/rohola/codes/transfer-learning-conv-ai/out/first_sentence/generated_sentences_emotion_recog.tsv", 'w') as fw:
    with open("/home/rohola/codes/transfer-learning-conv-ai/out/generated_sentences_logs_emotion_recog.tsv") as fr:
        for line in fr:
            text = line.split('\t')[0].rstrip()
            doc = nlp(text)
            try:
                first_sentence = list(doc.sents)[0]
                fw.write(first_sentence.text.rstrip() + "\tfear\n")
            except:
                first_sentence = "this is test."
                print("+")
                fw.write(first_sentence + "\tfear\n")