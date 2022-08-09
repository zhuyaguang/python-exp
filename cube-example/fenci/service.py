import bentoml
from bentoml.io import Text
import torch
import config
from bilstm_crf_token import readfile, token, idlist2tag

# config
word2id = readfile('BiLSTM_CRF_voc.txt')
tag2id = readfile('BiLSTM_CRF_tags.txt')



id2tag = dict((ids, tag) for tag, ids in tag2id.items())
device = config.device
runner = bentoml.pytorch.get('bilstm_crf:latest').to_runner()
svc = bentoml.Service("BiLSTM", runners=[runner])


@svc.api(input=Text(), output=Text())
def classify(input_series: str) -> str:
    word_list = []
    for i in input_series:
        word_list.append(i)
    # word_list = ['1', '9', '6', '2', '年', '1', '月', '出', '生', '，', '南', '京', '工', '学', '院', '毕', '业', '。']
    inputs = token([word_list], word2id, device)
    outputs = runner.__call__.run(inputs)
    tags = idlist2tag(outputs, tag2id, id2tag)    
    return ' '.join(tags[0])