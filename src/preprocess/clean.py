import re

def clean_hashtag(caption):
    return re.sub(r'\n?#\S+', '', caption).strip()

# remove the term To determnine if of the reasoning term
def clean_reasoning(reasoning):
    terms = reasoning.split(",")
    if ("To determine if" in terms[0]) or ("to determine if" in terms[0]):
        return ",".join(terms[1:]).strip()
    return reasoning