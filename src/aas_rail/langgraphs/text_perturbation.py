import random
import string
def lexical_noise(text, intensity=1):
    """
    Add lexical noise to a string using up to `distance` edits. `distance` depends on the length of the text and the intensity.
    Edits include: substitution, deletion, insertion, and transposition (swap).
    Swap is only applied if the index is not the last character.
    """
    max_distance = 0.1 * len(text)
    distance = int(max_distance * intensity)
    text = list(text)

    for _ in range(distance):
        if not text:
            operations = ["insertion"]
        else:
            operations = ["substitution", "deletion", "insertion"]
            if len(text) > 1:
                operations.append("swap")

        operation = random.choice(operations)
        index = random.randint(0, len(text) - 1) if text else 0

        if operation == "substitution":
            text[index] = random.choice(string.ascii_lowercase)

        elif operation == "deletion":
            del text[index]

        elif operation == "insertion":
            text.insert(index, random.choice(string.ascii_lowercase))

        elif operation == "swap":
            if index < len(text) - 1:
                text[index], text[index + 1] = text[index + 1], text[index]
            else:
                # fallback to substitution if swap isn't valid
                text[index] = random.choice(string.ascii_lowercase)

    return ''.join(text)



import random
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk import pos_tag

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None

def get_synonyms(word, pos=None):
    synonyms = set()
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, intensity=1):
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    num_replacements = int(0.1 * len(tokens) * intensity)

    spans = list(tokenizer.span_tokenize(text))
    tagged_tokens = pos_tag(tokens)

    eligible = []
    for i, (word, tag) in enumerate(tagged_tokens):
        if word.isalpha():
            wn_pos = get_wordnet_pos(tag)
            if wn_pos and get_synonyms(word, wn_pos):
                eligible.append((i, word, wn_pos))

    if not eligible:
        return text

    to_replace = random.sample(eligible, min(num_replacements, len(eligible)))
    replacements = {}

    for i, word, wn_pos in to_replace:
        synonyms = get_synonyms(word, wn_pos)
        if synonyms:
            replacements[i] = random.choice(synonyms)

    # Replace in original text using spans
    new_text = []
    last_idx = 0
    for i, (start, end) in enumerate(spans):
        new_text.append(text[last_idx:start])
        if i in replacements:
            new_text.append(replacements[i])
        else:
            new_text.append(text[start:end])
        last_idx = end
    new_text.append(text[last_idx:])

    return ''.join(new_text)


import spacy
from spacy.cli import download
import random

download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm") 

def degrade_grammar(text, intensity=1, pos_to_remove={"DET", "ADP", "AUX", "PUNCT"}):
    doc = nlp(text)
    removable = [token for token in doc if token.pos_ in pos_to_remove]
    num_to_remove = int(len(removable) * intensity)

    if num_to_remove > len(removable):
        num_to_remove = len(removable)

    tokens_to_remove = set(random.sample(removable, num_to_remove))
    spans_to_remove = sorted(tokens_to_remove, key=lambda token: token.idx, reverse=True)

    for token in spans_to_remove:
        start = token.idx
        end = start + len(token.text)

        # Handle trailing space more carefully
        if token.pos_ != "PUNCT":
            # Remove the space after the token if it exists
            if end < len(text) and text[end] == " ":
                end += 1
        else:
            # For punctuation: only remove space if it's not needed
            if end < len(text) and text[end] == " ":
                # Check if this punctuation is at end of sentence or followed by another punctuation
                if end + 1 == len(text) or not text[end + 1].isalnum():
                    end += 1  # OK to remove space

        text = text[:start] + text[end:]

    return text



import random
import re

DEFAULT_FILLERS = [
    "sort of", "maybe", "kind of", "or something", "you know", "perhaps",
    "more or less", "a little bit", "or related things", "I guess", "in a way",
    "at least to some extent", "if you will"
]

def ambiguate_text(text, intensity=1, fillers=None, seed=None):
    if seed is not None:
        random.seed(seed)
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    num_insertions = int(0.1 * len(tokens) * intensity)
    if fillers is None:
        fillers = DEFAULT_FILLERS

    # Find word boundaries where it's grammatically safer to insert filler
    candidates = []
    for match in re.finditer(r'\b(?:is|was|are|were|seems|looks|and|but|or|so)\b', text):
        start = match.end()
        # Only allow insertion if not at the end of the sentence
        if start < len(text) and text[start] == ' ':
            candidates.append(start + 1)  # Insert after the space

    # If not enough grammatical spots, add random safe spaces between words
    if len(candidates) < num_insertions:
        extra_spots = [m.start() + 1 for m in re.finditer(r'\s', text)]
        candidates += random.sample(extra_spots, min(num_insertions - len(candidates), len(extra_spots)))

    candidates = list(set(candidates))
    random.shuffle(candidates)
    insert_points = sorted(candidates[:num_insertions])

    degraded_text = text
    offset = 0
    for point in insert_points:
        filler = random.choice(fillers)
        insertion = filler + " "
        degraded_text = degraded_text[:point + offset] + insertion + degraded_text[point + offset:]
        offset += len(insertion)

    return degraded_text



import random
import spacy
from spacy.language import Language

# Load the small English model (run this once at program start)
nlp = spacy.load("en_core_web_sm")

# Custom sentence boundary rule
@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for token in doc[:-1]:
        # Disable sentence start after numbers followed by a period (common for lists)
        if token.like_num and doc[token.i + 1].text == ".":
            doc[token.i+2].is_sent_start = False
    return doc

# Add custom rule BEFORE existing sentence boundary logic
nlp.add_pipe("custom_sentencizer", before="parser")


def add_irrelevance(text, intensity=0.1, transitions=None, irrelevant_topics=None):
    """
    Adds contextually irrelevant information within the text by inserting irrelevant sentences
    after randomly selected sentences, based on the specified intensity.

    Uses spaCy for reliable sentence boundary detection.
    Returns exactly the original text if intensity == 0.0.

    Parameters:
        text (str): Original text to degrade.
        intensity (float): Proportion of sentences to modify (e.g., 0.1 = 10%).
        transitions (list): Optional list of transition phrases.
        irrelevant_topics (list): Optional list of irrelevant topics.

    Returns:
        str: Degraded text.
    """

    # if intensity == 0.0:
    #     return text

    if transitions is None:
        transitions = [
            "and also consider",
            "while thinking about",
            "as well as the effects of",
            "alongside",
            "bearing in mind",
            "not forgetting",
            "in the context of"
        ]

    if irrelevant_topics is None:
        irrelevant_topics = [
            "the importance of pineapple in modern diets",
            "the color blue in Renaissance paintings",
            "how cats perceive time",
            "weather patterns on Mars",
            "the mating habits of jellyfish",
            "quantum fluctuations in empty space",
            "why toast always lands butter-side down",
            "the cultural significance of clowns",
            "lava lamp physics"
        ]

    doc = nlp(text)
    sentence_spans = list(doc.sents)

    if not sentence_spans:
        return text

    num_sentences = len(sentence_spans)
    num_insertions = int(0.1*num_sentences * intensity)

    # if num_insertions == 0:
    #     return text

    # Select random sentence indices
    insertion_indices = random.sample(range(num_sentences), num_insertions)

    # Prepare (position, insertion_text) pairs
    insertions = []
    for idx in insertion_indices:
        span = sentence_spans[idx]
        end_pos = span.end_char
        transition = random.choice(transitions)
        topic = random.choice(irrelevant_topics)
        irrelevant_sentence = f" {transition.capitalize()} {topic}."
        insertions.append((end_pos, irrelevant_sentence))

    # Sort insertions in reverse to avoid index shifting
    insertions.sort(reverse=True)

    modified_text = text
    for pos, insertion_text in insertions:
        modified_text = modified_text[:pos] + insertion_text + modified_text[pos:]

    return modified_text


def degrade_text(text, intensity=1):
    """
    Degrades a prompt by applying various perturbations.
    """
    text = synonym_replacement(text, intensity)
    text = degrade_grammar(text, intensity)
    text = ambiguate_text(text, intensity)
    text = add_irrelevance(text, intensity)
    text = lexical_noise(text, intensity)
    return text