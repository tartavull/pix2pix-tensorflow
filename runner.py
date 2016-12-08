import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob
from model import pix2pix

class QueueRunner(object):
  def __init__(self, sess, args, filename_glob, batch_size=1, image_height=256, image_width=512, image_channels=3, image_type='jpeg'):
    #We create a tf variable to hold the global step, this has the effect
    #that when a checkpoint is created this value is saved.
    #Making the plots in tensorboard being continued when the model is restored.
    global_step = tf.Variable(0)
    increment_step = global_step.assign_add(1)


    #Create a queue that will be automatically fill by another thread
    #as we read batches out of it
    train_in = self.batch_queue(filename_glob, batch_size, image_height, image_width, image_channels, image_type)
    
    # create model
    model = pix2pix(sess, train_in, global_step, args, image_size=args.fine_size, batch_size=args.batch_size,
                            output_size=args.fine_size, dataset_name=args.dataset_name,
                            checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
    model.train(args)

    init_op = tf.global_variables_initializer()
    #This is required to intialize num_epochs for the filename_queue
    init_local = tf.local_variables_initializer()

    # Initialize the variables (like the epoch counter).
    sess.run([init_op,init_local])

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:

      progress = tqdm()
      while not coord.should_stop():
          # Run training steps or whatever
          global_step = sess.run(increment_step)
          progress.update()
          model.train_step()

    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


  def read_png(self, filename_queue, image_height, image_width, image_channels, image_type):
    reader = tf.WholeFileReader()
    _ , value = reader.read(filename_queue)
    if image_type == 'jpeg':
      raw_int_image = tf.image.decode_jpeg(value, channels=image_channels)
    else:
      raw_int_image = tf.image.decode_png(value, channels=image_channels)
    center_cropped_image = tf.image.resize_image_with_crop_or_pad(raw_int_image, image_height, image_width)
    float_image = tf.cast(center_cropped_image,tf.float32)
    float_image = tf.sub(tf.div(float_image, 127.5), 1.0)

    #required for graph shape inference
    float_image.set_shape((image_height,image_width,image_channels))

    return float_image

  def batch_queue(self, filenames, batch_size, image_height, image_width, image_channels, image_type, num_epochs=None):

    with tf.variable_scope("batch_queue"):
      filename_queue = tf.train.string_input_producer(
          glob(filenames), num_epochs=num_epochs, shuffle=True)
      image_node = self.read_png(filename_queue, image_height, image_width, image_channels, image_type)
      # min_after_dequeue defines how big a buffer we will randomly sample
      #   from -- bigger means better shuffling but slower start up and more
      #   memory used.
      # capacity must be larger than min_after_dequeue and the amount larger
      #   determines the maximum we will prefetch.  Recommendation:
      #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * batch_size
      example_batch = tf.train.shuffle_batch(
          [image_node], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
      return example_batch