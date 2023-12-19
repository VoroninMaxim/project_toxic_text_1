
from transformers import pipeline

clf = pipeline(task= 'sentiment-analysis', model='s-nlp/russian_toxicity_classifier')

text = ['Просто займись делом, и всё как рукой снимет.', 'Какая гадость эта ваша заливная рыба!',
        'Нет, не хочу ни говорить, ни слушать.', 'Не приписывай самому себе того, что не тебе принадлежит.']
#----------------------------
def data(text):
    for row in text:
        yield row
#----------------------------
for out in clf(data(text)):
    print(out)
