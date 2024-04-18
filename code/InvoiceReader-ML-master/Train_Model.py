import spacy
import json
from spacy.training import Example

#JSON data
with open('C:/Users/admin/Documents/imagetotext- python/invoicemodel/Invoice/InvoiceReader-ML-master/Training_data/training_data.json', 'r') as file:
    data = json.load(file)

#blank spaCy NER model
nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")

for item in data:
    annotations = item['annotation']
    for ent in annotations['entities']:
        ner.add_label(ent[2])

train_data = []
for entry in data:
    text = entry['content']
    entities = entry['annotation']['entities']
    train_data.append((text, {'entities': entities}))

#training parameters
n_iter = 10  
learn_rate = 0.001 
dropout = 0.5 

# Train the NER model
optimizer = nlp.begin_training()
optimizer.learn_rate = learn_rate 
for i in range(n_iter):
    spacy.util.fix_random_seed(1) 
    losses = {}
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], losses=losses, drop=dropout)
    print(losses)

            
nlp.to_disk("ner_model")