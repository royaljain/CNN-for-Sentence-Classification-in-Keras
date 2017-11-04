import numpy as np

dataPath = '/Users/royal/Desktop/MyDeepMoji/DeepMoji/data/WikiPersonnalAttack/'

train_x  = np.load(dataPath + 'train_x_personal_attack.npy')
train_y  = np.load(dataPath + 'train_y_personal_attack.npy')
dev_x  = np.load(dataPath + 'dev_x_personal_attack.npy')
dev_y  = np.load(dataPath + 'dev_y_personal_attack.npy')
test_x  = np.load(dataPath + 'test_x_personal_attack.npy')
test_y  = np.load(dataPath + 'test_y_personal_attack.npy')


train_size = len(train_x)



train_val_x = np.array(train_x.tolist() + dev_x.tolist())
train_val_y = np.array(train_y.tolist() + dev_y.tolist())


print len(train_val_x)
print len(train_val_y)

from data_helpers import load_data_and_labels, clean_str, pad_sentences, build_vocab, build_input_data


train_val_x = [clean_str(sent) for sent in train_val_x]
train_val_x = [s.split(" ") for s in train_val_x]
labels = train_val_y

sentences_padded, seq_len = pad_sentences(train_val_x)
vocabulary, vocabulary_inv_list = build_vocab(sentences_padded)
train_val_x, labels = build_input_data(sentences_padded, labels, vocabulary)

vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

train_x = train_val_x[:train_size]
dev_x = train_val_x[train_size:]

print len(train_x)
print len(dev_x)


train_y = train_val_y[:train_size]
dev_y = train_val_y[train_size:]


test_x = [clean_str(sent) for sent in test_x]
test_x = [s.split(" ") for s in test_x]
sentences_padded, seq_len = pad_sentences(test_x, seq_len)
test_x, labels = build_input_data(sentences_padded, labels, vocabulary)

print len(test_x)
print len(test_y)


def getTrain():
    return train_x, train_y, dev_x, dev_y, test_x, test_y, vocabulary_inv