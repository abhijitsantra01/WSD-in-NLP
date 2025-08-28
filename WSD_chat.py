import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

class WordSenseDisambiguator:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        
        # Ambiguous words to focus on
        self.ambiguous_words = {
            'bank', 'bark', 'bat', 'bear', 'bow', 'box', 'can', 'card', 'case', 'cast',
            'chair', 'charge', 'check', 'club', 'coach', 'code', 'cold', 'cook', 'cool',
            'course', 'court', 'cover', 'craft', 'crane', 'cross', 'crown', 'cut', 'deal',
            'deck', 'dish', 'dock', 'draw', 'drill', 'drive', 'drop', 'duck', 'face',
            'fair', 'fall', 'fan', 'fast', 'field', 'figure', 'file', 'fire', 'firm',
            'fish', 'fit', 'fix', 'flat', 'flight', 'floor', 'fly', 'fold', 'foot',
            'form', 'frame', 'free', 'game', 'glass', 'ground', 'hand', 'hard', 'head',
            'heat', 'hit', 'hold', 'house', 'iron', 'job', 'judge', 'key', 'kind',
            'land', 'last', 'lead', 'leaf', 'left', 'light', 'line', 'live', 'lock',
            'long', 'lot', 'mail', 'make', 'man', 'mark', 'match', 'matter', 'may',
            'mean', 'mine', 'miss', 'model', 'move', 'nail', 'name', 'net', 'note',
            'order', 'pack', 'page', 'palm', 'paper', 'park', 'part', 'pass', 'patient',
            'pick', 'picture', 'piece', 'pin', 'pitch', 'place', 'plan', 'plant', 'play',
            'point', 'pool', 'post', 'pound', 'press', 'program', 'pupil', 'quarter',
            'race', 'rail', 'range', 'rate', 'rest', 'right', 'ring', 'rock', 'roll',
            'room', 'root', 'round', 'row', 'run', 'safe', 'sail', 'scale', 'school',
            'score', 'seal', 'season', 'seat', 'second', 'sense', 'set', 'ship', 'shop',
            'shot', 'show', 'side', 'sign', 'sink', 'site', 'slip', 'smoke', 'snap',
            'sound', 'space', 'spare', 'spring', 'square', 'stage', 'stand', 'star',
            'state', 'stick', 'still', 'stock', 'store', 'story', 'strike', 'string',
            'suit', 'sum', 'switch', 'table', 'take', 'tape', 'task', 'team', 'term',
            'test', 'tie', 'time', 'tip', 'top', 'track', 'train', 'trip', 'turn',
            'type', 'walk', 'wall', 'watch', 'wave', 'way', 'well', 'wheel', 'will',
            'wind', 'wing', 'work', 'yard', 'year'
        }
        
        print("Word Sense Disambiguator initialized!")
        print(f"Tracking {len(self.ambiguous_words)} ambiguous words")

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens

    def get_wordnet_pos(self, word):
        """Get WordNet POS tag"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
        return tag_dict.get(tag, wn.NOUN)

    def get_context_vector(self, sentence, target_word):
        """Create context vector from sentence excluding target word"""
        tokens = self.preprocess_text(sentence)
        context_words = [word for word in tokens if word != target_word.lower()]
        return ' '.join(context_words)

    def get_synset_definition_vector(self, synset):
        """Get definition and examples as context for synset"""
        definition = synset.definition()
        examples = ' '.join(synset.examples())
        return f"{definition} {examples}"

    def disambiguate_word(self, word, sentence):
        """Disambiguate a single word in context"""
        word_lower = word.lower()
        
        pos = self.get_wordnet_pos(word)
        synsets = wn.synsets(word_lower, pos=pos)
        
        if not synsets:
            return {
                'word': word,
                'best_sense': 'No synsets found',
                'definition': 'Word not found in WordNet',
                'confidence': 0.0
            }
        
        if len(synsets) == 1:
            synset = synsets[0]
            return {
                'word': word,
                'best_sense': synset.name(),
                'definition': synset.definition(),
                'confidence': 1.0
            }
        
        context = self.get_context_vector(sentence, word)
        
        if not context.strip():
            synset = synsets[0]
            return {
                'word': word,
                'best_sense': synset.name(),
                'definition': synset.definition(),
                'confidence': 0.5
            }

        synset_contexts = []
        for synset in synsets:
            synset_context = self.get_synset_definition_vector(synset)
            synset_contexts.append(synset_context)

        all_contexts = [context] + synset_contexts
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_contexts)
            context_vector = tfidf_matrix[0:1]
            synset_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(context_vector, synset_vectors)[0]

            best_idx = np.argmax(similarities)
            best_synset = synsets[best_idx]
            confidence = float(similarities[best_idx])
            
            return {
                'word': word,
                'best_sense': best_synset.name(),
                'definition': best_synset.definition(),
                'confidence': confidence
            }
            
        except Exception as e:
            synset = synsets[0]
            return {
                'word': word,
                'best_sense': synset.name(),
                'definition': synset.definition(),
                'confidence': 0.3
            }

    def process_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        results = []
        
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.ambiguous_words:
                result = self.disambiguate_word(token, sentence)
                results.append(result)
        
        return results

    def format_results(self, results):
        if not results:
            return "No ambiguous words found in the sentence."
        
        formatted = "\nWord Sense Analysis:\n" + "="*40 + "\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"\n{i}. Word: '{result['word']}'\n"
            formatted += f"   Best sense: {result['best_sense']}\n"
            formatted += f"   Definition: {result['definition']}\n"
            formatted += f"   Confidence: {result['confidence']:.3f}\n"
            formatted += "-" * 30 + "\n"
        
        return formatted

class WSDChatBot:
    def __init__(self):
        self.disambiguator = WordSenseDisambiguator()
        print("\nWord Sense Disambiguation ChatBot Ready!")
        print("I can help disambiguate word meanings in your sentences.")
        print("Type 'quit' to exit.\n")

    def chat(self):
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("ChatBot: Goodbye! Thanks for using the chatbot!")
                    break
                
                if not user_input:
                    print("ChatBot: Please enter a sentence to analyze.")
                    continue
                
                print("\nChatBot: Analyzing your sentence...\n")
                
                # Process the sentence
                results = self.disambiguator.process_sentence(user_input)
                
                # Format and display results
                response = self.disambiguator.format_results(results)
                print("ChatBot:", response)
                
            except KeyboardInterrupt:
                print("\n\nChatBot: Goodbye!")
                break
            except Exception as e:
                print(f"ChatBot: Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    print("Word Sense Disambiguation System")
    chatbot = WSDChatBot()
    chatbot.chat()
