import os
import random
import tensorflow as tf
import math
import sys
_NUM_VALIDATION=350
_RANDOM_SEED=0
_NUM_SHARDS=5
data_dir='F:\\DL\\tfrecords'
LABELS_FILENAME = 'labels.txt'
class ImageReader(object):
    def __init__(self):
        self._decode_jpeg_data=tf.placeholder(dtype=tf.string)
        self._decode_jpeg=tf.image.decode_jpeg(self._decode_jpeg_data,channels=3)

    def read_image_dims(self,sess,image_data):
        image=self.decode_jpeg(sess,image_data)
        return image.shape[0],image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values=[values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
    if not isinstance(values,(tuple,list)):
        values=[values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def image_to_tfexample(image_data,image_format,height,width,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

def _get_filenames_and_classes(data_dir):
    flower_root=os.path.join(data_dir,'my_data')
    directories=[]
    class_names=[]
    for filename in os.listdir(flower_root):
        path=os.path.join(flower_root,filename)
        if os.path.isdir(path):
            directories.append(path) #directories为每种类别文件夹的绝对路径
            class_names.append(filename) #类别名

    photo_filenames=[]
    for diretory in directories:
        for filename in os.listdir(diretory):
            path=os.path.join(diretory,filename) #path为每一张图片的绝对路径
            photo_filenames.append(path)

    return photo_filenames,sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train','validation']

    num_per_shard=int(math.ceil(len(filenames)/float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
           for shard_id in range(_NUM_SHARDS):
               output_filename=_get_dataset_filename(dataset_dir,split_name,shard_id)

               with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                   start_ndx=shard_id*num_per_shard
                   end_ndx=min((shard_id+1)*num_per_shard,len(filenames))
                   for i in range(start_ndx,end_ndx):
                       sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                       sys.stdout.flush()

                       image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                       height, width = image_reader.read_image_dims(sess, image_data)

                       class_name = os.path.basename(os.path.dirname(filenames[i]))
                       class_id = class_names_to_ids[class_name]

                       example = image_to_tfexample(
                           image_data, b'jpg', height, width, class_id)
                       tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def write_label_file(labels_to_class_names,dataset_dir,filename=LABELS_FILENAME):
    labels_filename=os.path.join(dataset_dir,filename)
    with open(labels_filename,'w') as f:
        for label in labels_to_class_names:
            class_name=labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

photo_filenames,class_names=_get_filenames_and_classes(data_dir)
class_names_to_ids=dict(zip(class_names,range(len(class_names))))

#分割成训练集和验证集
random.seed(_RANDOM_SEED)
#shuffle
random.shuffle(photo_filenames)
training_filenames = photo_filenames[_NUM_VALIDATION:]
validation_filenames = photo_filenames[:_NUM_VALIDATION]

_convert_dataset('train', training_filenames, class_names_to_ids,
                 data_dir)
_convert_dataset('validation', validation_filenames, class_names_to_ids,
                 data_dir)

# Finally, write the labels file:
labels_to_class_names=dict(zip(range(len(class_names)),class_names))
write_label_file(labels_to_class_names,data_dir)
print('\nFinished converting the Flowers dataset!')




