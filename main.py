import sys
import os
import json
import spacy
from simpleneighbors import SimpleNeighbors
from flask import Flask, request
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5000))

nlp = spacy.load("en_core_web_md")

def load_words():
    with open("words_alpha.txt") as word_file:
        valid_words = list(word_file.read().split())
    return valid_words

all_words = load_words()
# all_words = list(nlp.vocab.strings)

def vec(word):
    return nlp(word, disable=["parser", "tagger", "ner"]).vector

embeddings = [vec(w) for w in all_words]

lookup = SimpleNeighbors(300)
for v, w in zip(embeddings, all_words):
    lookup.add_one(w, v)
lookup.build()

def nearest_words(word, used_words):
    ws = [w for w in lookup.nearest(vec(word), 156) if w not in used_words][:5]
    used_words.extend(ws)
    return ws

def get_words(word):
    words = dict()
    used_words = list()
    used_words.append(word)
    words["word"] = word
    words["children"] = list()
    children_words1 = nearest_words(word, used_words)
    for w1 in children_words1:
        child1 = dict()
        child1["word"] = w1
        child1["children"] = list()
        children_words2 = nearest_words(w1, used_words)
        for w2 in children_words2:
            child2 = dict()
            child2["word"] = w2
            child2["children"] = list()
            children_words3 = nearest_words(w2, used_words)
            for w3 in children_words3:
                child3 = dict()
                child3["word"] = w3
                child3["children"] = list()
                child2["children"].append(child3)
            child1["children"].append(child2)
        words["children"].append(child1)
    return words

@app.route("/api/v1/flask/data", methods=["POST"])
def postdata():
    data = request.get_json()
    word = data["word"]
    print("received word:", word)
    words = get_words(word)
    return json.dumps(words)

if __name__ == "__main__":
    print("Listening on port:", PORT)
    http_server = WSGIServer(("", PORT), app)
    http_server.serve_forever()
