import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from data.twitter import data
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')                   # Twitter


names = ['javier', 'alvaro', 'daniel', 'alfonso', 'adrian', 'jaime', 'edu', 'suso', 'jesus',
         'miriam', 'guillermo', 'pablo', 'fer', 'manu', 'dani', 'ignacio', 'juanjo', 'lucas',
         'diego', 'escarcha', 'frost']

# print("metadata", metadata, idx_q, idx_a)
print("idx_q",idx_q.shape)
print("idx_a",idx_a.shape)

(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

xseq_len = len(trainX)#.shape[-1]
yseq_len = len(trainY)#.shape[-1]
assert xseq_len == yseq_len
batch_size = 32
n_step = int(xseq_len/batch_size)
xvocab_size = len(metadata['idx2w']) # 8002 (0~8001)
emb_dim = 1024

w2idx = metadata['w2idx']   # dict  word 2 index
idx2w = metadata['idx2w']   # list index 2 word
for name in names:
    if name in w2idx:
        print("Name", name, "is in w2idx")


start_id = xvocab_size  # 8002
end_id = xvocab_size+1  # 8003

w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
unk_word = "unk"

idx2w = idx2w + ['start_id', 'end_id']

xvocab_size = yvocab_size = xvocab_size + 2

""" A data for Seq2Seq should look like this:
input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
target_mask : [1, 1, 1, 1, 0]
"""

print("encode_seqs", [idx2w[id] for id in trainX[10]])
target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
print("target_seqs", [idx2w[id] for id in target_seqs])
decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
print("decode_seqs", [idx2w[id] for id in decode_seqs])
target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
print("target_mask", target_mask)
print(len(target_seqs), len(decode_seqs), len(target_mask))

###============= model
def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    emb_dim = 1024
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True) # remove if TL version == 1.8.0+
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
        net_rnn = Seq2Seq(net_encode, net_decode,
                          cell_fn = tf.contrib.rnn.BasicLSTMCell,
                          n_hidden = emb_dim,
                          initializer = tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode = None,
                          dropout = (0.5 if is_train else None),
                          n_layer = 3,
                          return_seq_2d = True,
                          name = 'seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
    return net_out, net_rnn


# model for inferencing
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
net, net_rnn = model(encode_seqs2, decode_seqs2, is_train=False, reuse=False)
y = tf.nn.softmax(net.outputs)

sess = tf.InteractiveSession()

tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name='n.npz', network=net)


# seeds = ["we are having an epic thursday today at epic",
#          "looking forward to block chain breakfast tomorrow",
#          "fantastic talk yesterday about git with javier",
#          "yaw warning in men's bathroom",
#          "please be careful and use the toilet brush",
#          "i loved machine learning tech talk by alvaro and escarcha",
#          "i think rajoy is a good president"]
# for seed in seeds:
#     print("Query >", seed)
#     seed_id = []
#     for w in seed.split(" "):
#         if w in w2idx:
#             seed_id.append(w2idx[w])
#         else:
#             if w in names:
#                 seed_id.append(w2idx['jesus'])
#             elif w == 'rajoy':
#                 seed_id.append(w2idx['trump'])
#             else:
#                 seed_id.append(w2idx[unk_word])
#     # seed_id = [w2idx[w] for w in seed.split(" ")]
#     for _ in range(5):  # 1 Query --> 5 Reply
#         # 1. encode, get state
#         state = sess.run(net_rnn.final_state_encode,
#                          {encode_seqs2: [seed_id]})
#         # 2. decode, feed start_id, get first word
#         #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
#         o, state = sess.run([y, net_rnn.final_state_decode],
#                             {net_rnn.initial_state_decode: state,
#                              decode_seqs2: [[start_id]]})
#         w_id = tl.nlp.sample_top(o[0], top_k=3)
#         w = idx2w[w_id]
#         # 3. decode, feed state iteratively
#         sentence = [w]
#         for _ in range(30): # max sentence length
#             o, state = sess.run([y, net_rnn.final_state_decode],
#                                 {net_rnn.initial_state_decode: state,
#                                  decode_seqs2: [[w_id]]})
#             w_id = tl.nlp.sample_top(o[0], top_k=2)
#             w = idx2w[w_id]
#             if w_id == end_id:
#                 break
#             sentence = sentence + [w]
#         print(" >", ' '.join(sentence))

def answer(message):
    seed_id = []
    unknown_words = []
    for w in message.split(" "):
        w = w.lower()
        if w.endswith(".") or w.endswith(",") or w.endswith("?") or w.endswith("!"):
            print("Character removed!", w, w[:-1])
            w = w[:-1]

        if w in w2idx:
            seed_id.append(w2idx[w])
        else:
            if w in names:
                seed_id.append(w2idx['jesus'])
            elif w == 'rajoy':
                seed_id.append(w2idx['trump'])
            else:
                unknown_words.append(w)
                seed_id.append(w2idx[unk_word])

    sentences = []
    for _ in range(5):  # 1 Query --> 5 Reply
        # 1. encode, get state
        state = sess.run(net_rnn.final_state_encode,
                         {encode_seqs2: [seed_id]})
        # 2. decode, feed start_id, get first word
        #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
        o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                             decode_seqs2: [[start_id]]})
        w_id = tl.nlp.sample_top(o[0], top_k=3)
        w = idx2w[w_id]
        # 3. decode, feed state iteratively
        sentence = [w]
        for _ in range(30): # max sentence length
            o, state = sess.run([y, net_rnn.final_state_decode],
                                {net_rnn.initial_state_decode: state,
                                 decode_seqs2: [[w_id]]})
            w_id = tl.nlp.sample_top(o[0], top_k=2)
            w = idx2w[w_id]
            if w_id == end_id:
                break
            sentence = sentence + [w]
        sentences.append(' '.join(sentence))
    return sentences, unknown_words

@app.route('/')
def index():
    return render_template('main.html')


@app.route('/answer', methods=['POST'])
def answer_request():
    print("Request message")
    json_body = request.get_json(force=True)
    print("Request message json", json_body)
    response, unkown_words = answer(json_body["message"])
    print("Request message response", response)
    print("Unkown words", unkown_words)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)