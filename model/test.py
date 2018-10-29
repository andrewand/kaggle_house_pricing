from data_load import *

data = load_raw_data()
train = data.loc[data['data_type'] == 'train']
train = train.drop(['data_type'], axis=1)
train.shape