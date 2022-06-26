import pandas as pd

PRONOUNS = ['i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'his', 'hers', 'my', 'mine', 'our', 'ours', 'your', 'yours', 'their', 'theirs',
            'thy', 'thee', 'thou']

HELPING_VERBS = ['am', 'is', 'are', 'was', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
                 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could']

# with open('/Users/sunyambagga/Desktop/txtLAB-2/minimal-narrativity/data/places_lexicon.txt', 'r') as F:
#     PLACES_VOCAB = F.read().splitlines()
#     print("PLACE LEXICON has {} entries.".format(len(PLACES_VOCAB)))

# Concreteness:
# MAP_WORD_CONC = dict(pd.read_csv('../../Concreteness_ratings_Brysbaert_et_al_BRM.txt', delimiter='\t')[['Word', 'Conc.M']].values)
