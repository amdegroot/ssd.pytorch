import pandas as pd
from helperFunctions.joinHelpers import anti_join

ids = pd.read_csv('./data/bhjc20180123_bball/bhjc_trainval.txt', dtype='str', header=None)
test_set = ids.sample(n=50)

train_set = anti_join(ids, test_set)

test_set.to_csv('./data/bhjc20180123_bball/bhjc_testonly.txt', index=False)
train_set.to_csv('./data/bhjc20180123_bball/bhjc_trainonly.txt', index=False)
