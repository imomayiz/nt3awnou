""" 
Problem: 
Nt3awnou's platform collects raw data filled manually by users (people in need). 
Among this data is the user's localisation.
The localisation is a text input that is not standardized: 
 i.e. a user can input a single or multiple locations 
 (either douars/provinces/communes/regions or all combined), 
 in arabic or latin, with misspellings etc. 
This doesn't help in visualization or in statistics
where localisations can be redundant because they were written in different manners.

Examples
```
دوار تجكَالت
ابرداتن ازكور
خزامة
Tansgharte
دوار امندار
Douar Essour Tidrara Aghwatim Tahnaouet Al Haouz
دوار تكاديرت
Douar Essour tidrara- aghouatine- Tahanaout-El Haouz
```
Solution:
We collected a reference dataset that contains all douar names (arabic and latin)
with their corresponding regions, communes and provinces.
We developed methods using fuzzy matching and phonetics
to map the user's localisation to the closest match in the reference dataset

"""

from typing import Tuple
from pyphonetics import RefinedSoundex, Metaphone
import math
import difflib
import re


EPICENTER_LOCATION = [31.12210171476489, -8.42945837915193]
certainty_threshold = 1


def extract_ngrams(text, n):
    """ 
    A function that returns a list of n-grams from a text
    """
    ngrams = []

    if n < 1 or n > len(text):
        return ngrams  # Return an empty list if n is invalid

    # Iterate through the text and extract n-grams
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngrams.append(' '.join(ngram))

    return ngrams


def get_phonetics_distance(w1, w2):
    """
    A function that calculates levenhstein distance between phonetics
    representation of two words: add error term to the score
    """
    rs = RefinedSoundex()
    mt = Metaphone()
    d1 = mt.distance(w1, w2, metric='levenshtein')
    d2 = rs.distance(w1, w2, metric='levenshtein')
    res = (d1 + d2) / 2 + 0.05
    return res 


def get_top_n_phonetics_matches(
    w: str, ref_words: list, threshold=1, top_n=1) -> list[Tuple]:
  """
  A function that returns the top_n closest words to w from ref_words
  for which distance <= threshold
  using phonetical representation
  """
  if not w:
    return list()
  distances = {x: get_phonetics_distance(w, x) for x in ref_words}
  selected_words = {x: d for x, d in distances.items() if d<=threshold}
  sorted_d = dict(sorted(selected_words.items(), key=lambda item: item[1]))

  return list(sorted_d.items())[:top_n]


def get_geometric_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """ 
    A function that returns the distance between two points on earth
    using the haversine formula
    """
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a0 = (math.sin(dlat / 2)) ** 2 + math.cos(math.radians(lat1))
    a = a0 * math.cos(math.radians(lat2)) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c
    return distance


def are_village_names_similar(village_a: str, village_b: str) -> float:
    """ 
    A function that returns True if the two villages 
    are similar using strict fuzzy matching
    """
    if difflib.SequenceMatcher(None, village_a, village_b).ratio() >= 0.90:
        return True
    return False


def get_uncertainty_range(input_dict: dict, threshold: float) -> list:
    """ 
    A function that returns a list of tuples of the closest matches
    """
    if len(input_dict)<=1:
        return input_dict

    # sort by distance
    sorted_items = sorted(input_dict.items(), key=lambda item: item[1][1])
    data = {key: value for key, value in sorted_items}

    # Iterate through the keys in the dictionary
    keys = list(data.keys())
    min_key = keys[0]
    min_value = data[min_key][1]

    # Initialize a list to store the result tuples
    result = {f"{min_key}":data[min_key]}

    for j in range(1, len(keys)):
        key2 = keys[j]
        value2 = data[key2][1]

        # Calculate the absolute difference between the float values
        difference = abs(min_value - value2)

        # If the difference is less than the threshold, add the tuple to the result
        if difference <= threshold:
            result[key2] = data[key2]
        else:
            break

    return result


def match_word(w, ref_dict, select_one_match=False):
  """ 
  A function that returns the closest match of w from ref_dict
  using phonetical representation and fuzzy matching
  """
  w = w.strip().upper()

  if len(w)==0:
    return {}

  else:
    closest_ref_w = dict()
    use_phonetics = True

    for category, names in ref_dict.items():
      # check exact matching
      if w in names:
        use_phonetics = False
        closest_ref_w[category] = (w, 0)
        break

      # check textual similarity (fuzzy matching)
      sim = list(map(lambda x:are_village_names_similar(w,x), names))
      similar_names = [names[i] for i in range(len(names)) if sim[i]==True]
      if similar_names:
        use_phonetics = False
        closest_ref_w[category] = (similar_names[0], 0.01) if select_one_match else list(map(lambda x:(x, 0.01), similar_names))

      # if no similar name was found check phonetical similarity
      else:
        res = get_top_n_phonetics_matches(w, names, threshold=2, top_n=1)
        if res:
          closest_ref_w[category] = res[0] # get closest match

    if closest_ref_w and use_phonetics:
      if not select_one_match:
        closest_ref_w = get_uncertainty_range(closest_ref_w, certainty_threshold)
      else:
        k, v = min(closest_ref_w.items(), key=lambda x: x[1][1])
        closest_ref_w = {k: v}

  return closest_ref_w


def parse_and_map_localisation(text: str, ref_dict: dict, select_one_match: bool=True):
  """ 
  A function that parses text containing users localisation
  and returns the closest matches per categoty from ref_dict
  Example: 
    input = COMMUNE MZODA : DOUARS : TOUKHRIBIN –TLAKEMT - COMMUNE IMINDOUNITE : DOUAR AZARZO
    output = {'commune_fr': ('IMINDOUNIT', 0.01), 'nom_fr': ('TOUKHRIBINE', 0.01)}
  """
  toxic = r"\bدوار|مصلى|\(|\)|douars?|communes?|cercles?|provinces?|villes?|regions?|caidate?|and|جماعة|\b|:|-|\d"
  text = re.sub(toxic, '', text.lower())
  regex_pattern = r"\|| |\.|,|/|et |و "
  tokens = re.split(regex_pattern, text.replace('-', ' '))
  filtered_tokens = [s for s in tokens if s.strip() != '']

  ngrams_mapping = {}

  for n in range(1, len(filtered_tokens)+1):

    # generate ngrams
    ngrams = extract_ngrams(filtered_tokens, n)

    # init dict with ngram mapping
    mapping_ngram = {}

    # generate a mapping for the ngram with argmin matches
    for tok in ngrams:
      res = match_word(tok, ref_dict, select_one_match=select_one_match)
      if not res:
        continue

      min_k, min_v = min(res.items(), key=lambda x:x[1][1])

      # if min_k in previous tokens, then choose the min, else add it to mapping
      if min_k in mapping_ngram:
        saved_match, saved_distance = mapping_ngram[min_k]

        if saved_distance > min_v[1]:
          mapping_ngram[min_k] = min_v

        else:
          continue

      else:
        mapping_ngram[min_k] = min_v

    ngrams_mapping[n] = mapping_ngram


  # first squeeze dict s.t. one match remains per category
  categories = ref_dict.keys()
  result = {}
  for _, inner_dict in ngrams_mapping.items():
    for k in categories:
        # Check if the key exists in the inner dictionary
        if k in inner_dict:
          current_match, current_val = inner_dict[k]
          if k in result:
            previous_match, previous_val = result[k]
            if current_val < previous_val:
              result[k] = (current_match, current_val)
          else:
            result[k] = (current_match, current_val)

  # then, discard matches with a high distance from min (set 0.5+min_d as threshold)
  thresh = min(result.values(), key=lambda x:x[1])[1] + 0.5
  output = {k: v_d for k, v_d in result.items() if v_d[1]<=thresh}

  return output