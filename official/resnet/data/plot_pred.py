import numpy as np
import pickle
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image, ImageFont, ImageDraw

VOCAB_PKL_PATH = 'vocab_short_list.pkl'
vocab = pickle.load(open(VOCAB_PKL_PATH, 'rb'))
if len(vocab) == 2:
    vocab = vocab[0]
    vocab = list(vocab)

vocab = np.array(vocab)
_NUM_CLASSES = 87
assert _NUM_CLASSES == len(vocab)

PRED_PKL_PATH = '/home/syalaman/tmp/preds.pkl'
preds = pickle.load(open(PRED_PKL_PATH, 'rb'))

DATA_PATH = glob.glob('scratch/VALIDATION/*')
DATA_PATH = sorted(DATA_PATH)

OUT_PATH = 'results'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

#font = ImageFont.truetype('Roboto-Bold.ttf', size=25)
(x1, y1) = (10, 10)
(x2, y2) = (150, 10)
color = 'rgb(255, 0, 0)'

with tf.Session() as sess:
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
            default_value=''),
        'image/ingrs': tf.VarLenFeature(dtype=tf.int64)
    }

    filename_queue = tf.train.string_input_producer(DATA_PATH, num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature_map)

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)

    label = tf.sparse_to_indicator(features['image/ingrs'], _NUM_CLASSES)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    count = 0
    try:
        while True:
            dish, ingrs = sess.run([image, label])
            #plt.imshow(dish)
            #plt.savefig(os.path.join(OUT_PATH, '%04d.jpg'%(count)))
            #plt.close()
            pred = np.array(preds[count]['classes'], dtype=bool)
            #with open(os.path.join(OUT_PATH, '%04d.txt'%(count)), 'w') as f:
            #    f.write(', '.join(vocab[pred]) + '\n')
            #    f.write(', '.join(vocab[ingrs]))
            im = Image.fromarray(dish)
            draw = ImageDraw.Draw(im)
            draw.text((x1, y1), '\n'.join(vocab[pred]), fill=color)#, font=font)
            draw.text((x2, y2), '\n'.join(vocab[ingrs]), fill=color)#, font=font)
            im.save(os.path.join(OUT_PATH, '%04d.jpg'%(count)))

            print count
            count += 1
    except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)
