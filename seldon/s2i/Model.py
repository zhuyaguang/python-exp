import pickle

with open('./ckpts/model.pkl','rb') as f:
    model = pickle.load(f)
sample = '王明是研究院的院长，出生与2022年'
pred_tags, prediction = model.predict(sample)
print(pred_tags,prediction)