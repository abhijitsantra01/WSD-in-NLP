import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Training data
sentences = [
    "He deposited money in the bank.",                   
    "She went to the bank to apply for a loan.",         
    "The river overflowed near the bank.",              
    "Children played by the river bank.",               
    "There is a new bank opening in our town.",          
    "Birds nested in the trees on the river bank.",     

    ("The river overflowed near the bank."),
    ("Children played by the river bank."),
    ("Birds nested in trees on the river bank."),
    ("The fisherman sat on the bank with his rod."),
    ("He walked along the bank watching the water flow."),
    ("The canoe drifted close to the bank."),

    "I took a loan from the bank.",                    
    "She works at the central bank.",                  
    "He picnicked near the river bank.",               
    "The bank was eroded by the flood waters.",

    "He withdrew money from the bank.",
    "The bank closed my credit card account.",
    "She deposited her paycheck at the bank.",
    "They opened a joint account at the bank.",
    "The bank charged a late fee.",
    "He went to the bank to apply for a mortgage.",
    "I set up a new account with an online bank.",
    "The bank approved my home loan.",
    "There is an ATM near the bank.",
    "She spoke with a bank advisor about retirement.",
    "The bank was robbed early this morning.",
    "He took a loan from a microfinance bank.",
    "The bank offers student loan refinancing.",
    "They visited the bank branch downtown.",
    "Interest rates at that bank are too high.",

    # River bank
    "They sat on the bank watching the river flow.",
    "A beaver built a dam near the bank.",
    "She slipped and fell on the muddy bank.",
    "The boat gently hit the bank.",
    "We camped overnight on the river bank.",
    "Wildflowers grew along the bank.",
    "He walked his dog beside the bank.",
    "The fisherman cast his line from the bank.",
    "Birds nested in the reeds by the bank.",
    "The flood destroyed homes close to the bank.",
    "Children played near the bank of the creek.",
    "The bank was eroded by the heavy rains.",
    "We watched ducks paddling close to the bank.",
    "The hiker rested on the bank of the stream.",
    "He tied the boat to a tree near the bank."
]
labels = [0, 0, 1, 1, 0,
    1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 
    1,0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1]  # 0 = financial, 1 = river

# Function to get BERT [CLS] token embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Prepare training features
X = [get_bert_embedding(s) for s in sentences]

# Train the classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

# Prediction function for new sentence
def predict_bank_sense(sentence):
    if "bank" not in sentence.lower():
        return "Please include the word 'bank' in your sentence."

    vector = get_bert_embedding(sentence)
    pred = clf.predict([vector])[0]
    meaning = "river bank 🌊" if pred == 1 else "financial bank 🏦"
    return f"Sentence: \"{sentence}\"\nPredicted sense of 'bank': {meaning}\n"


while True:
    user_input = input("Enter a sentence with the word 'bank' (or type 'exit' to quit):\n> ")
    if user_input.lower() == "exit":
        break
    print(predict_bank_sense(user_input))
