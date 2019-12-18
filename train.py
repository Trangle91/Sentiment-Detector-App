
import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("yelp_combined.txt", names=['sentence', 'label'], sep='\t')

#spliting data
X = df['sentence']
y = df['label']
X[0], y[0]

#cleaning text
import re
def clean_X(text):
  text = text.split(".")
  text = ' '.join(text)
  text = text.lower()
  text = re.sub("[^a-zA-Z\- ]"," ",text)
  text = re.sub(" +"," ",text)
  text = text.strip()
  return text

X = X.apply(lambda text: clean_X(text))

# Creating vocab
vocab = []
for sen in X:
  sen = sen.lower().split(" ")
  vocab.extend(sen)
vocab = list(set(vocab))
vocab.append("<PAD>")
vocab_size = len(vocab)
print("Vocab :: ",vocab)
print("Vocab size ::",vocab_size)

# Find sentence with maximum words and find the number of words present in it
max_len_sentence = None 
max_len = 0 
for sen in X:
  no_of_words = len(sen.split())
  if no_of_words>max_len:
    max_len = no_of_words
    max_len_sentence = sen
max_len_sentence, max_len

# <PAD>
X = [X[idx].split() + ["<PAD>"] * (max_len - len(X[idx].split()) ) for idx in range(len(X))]

# Converting all samples to indices
word_to_id = {vocab[word_id]:word_id for word_id in range(len(vocab))}
numerized_sentences = []
for sen in X:
  sen = [word_to_id[word] for word in sen]
  numerized_sentences.append(sen)

print("Numerized data :",numerized_sentences)

np.array(numerized_sentences).shape

dimension = 50
embedding_matrix = np.random.rand(vocab_size, dimension)
embedding_matrix

#creating vector matrices 
tf.reset_default_graph()
tf_embedding_matrix = tf.Variable(embedding_matrix)

sen_indices = tf.placeholder(dtype=tf.int32, shape=[None,None])
sen_to_vec = tf.nn.embedding_lookup(tf_embedding_matrix, sen_indices)
rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=16)
sen_to_vec = tf.cast(sen_to_vec, tf.float32)
outputs_for_tagging, output_for_classification = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=sen_to_vec, dtype=tf.float32)

classification_logits = tf.squeeze(tf.contrib.layers.fully_connected(output_for_classification, 1, activation_fn=tf.sigmoid))
classification_targets = tf.placeholder(dtype=tf.float32 ,shape=[None])
classification_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=classification_targets, logits=classification_logits))
train_op = tf.train.AdamOptimizer(0.01).minimize(classification_loss)

# """check accuracy"""
# correct_prediction = tf.equal(tf.argmax(classification_logits,1), tf.argmax(classification_targets,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(50):
    feed_dict = {sen_indices: numerized_sentences, classification_targets: y}
    output_classification, cl_logits, loss, _ = sess.run([classification_targets,
                                                                        classification_logits,
                                                                        classification_loss,
                                                                        train_op], feed_dict=feed_dict)
    print("Rnn final outputs:: ", output_classification)
    print("Loss :: ",loss)
    # print("Accuracy :: ",acc)
