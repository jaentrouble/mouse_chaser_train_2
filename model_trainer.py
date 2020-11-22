import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
import io
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import cv2
from pathlib import Path
import os
import pickle

class ChaserModel(keras.Model):
    """ChaserModel
    Gets an image and returns a heatmap

    Input
    -----
    input : tf.Tensor
        image tensor. Expects it to be tf.uint8 i.e. raw image
    
    Output
    ------
    heatmaps : dict of heatmaps
        {'name' : tf.Tensor}
    """
    def __init__(self, inputs, backbone_f, specific_fs):
        """
        Arguments
        ---------
        inputs: keras.Input

        backbone_f: function used universally across multiple outputs
        
        specific_fs: dict {'name' : model_function}
        """
        super().__init__()
        backbone_out = backbone_f(inputs)
        outputs = {}
        for out_name, sf in specific_fs.items():
            outputs[out_name]=(sf(backbone_out, out_name))
        self.heatmaps = keras.Model(inputs=inputs, outputs=outputs)
        self.heatmaps.summary()
        
    def call(self, inputs, training=None):
        casted = tf.cast(inputs, tf.float32) / 255.0
        return self.heatmaps(inputs, training=training)

class AugGenerator():
    """An iterable generator that makes augmented mouserec data

    NOTE: 
        Every img is reshaped to img_size

    return
    ------
    X : np.array, dtype= np.uint8
        shape : (HEIGHT, WIDTH, 3)
        color : RGB
    Y : dictionary of heatmaps
        {'name' : (HEIGHT, WIDTH, 1)}
    """
    def __init__(self, data_dir, class_labels, img_size):
        """ 
        arguments
        ---------
        data_dir : str
            path to the data directory (which has pickled files)
        class_labels : list of str
            list of classes to extract data
        img_size : tuple
            Desired output image size
            IMPORTANT : (HEIGHT, WIDTH)
        """
        self.class_labels = class_labels
        self.img_size = img_size
        self.data_dir = Path(data_dir)
        self.raw_data = []
        for pk_name in os.listdir(self.data_dir):
            with open(self.data_dir/pk_name,'rb') as f:
                self.raw_data.extend(pickle.load(f))
        
        self.n = len(self.raw_data)
        self.output_size = img_size
        self.aug = A.Compose([
            # A.OneOf([
            #     A.RandomGamma((40,200),p=1),
            #     A.RandomBrightness(limit=0.5, p=1),
            #     A.RandomContrast(limit=0.5,p=1),
            #     A.RGBShift(40,40,40,p=1),
            #     A.Downscale(scale_min=0.25,scale_max=0.5,p=1),
            #     A.ChannelShuffle(p=1),
            # ], p=0.8),
            # A.InvertImg(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=1),
            A.Resize(img_size[0], img_size[1]),
            # A.Cutout(8,img_size[0]//12,img_size[1]//12)
        ],
        # Unify all points order to 'ij' format i.e. 'yx' format
        keypoint_params=A.KeypointParams(format='yx',label_fields=['class_labels'])
        )

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.n)
        datum = self.raw_data[idx]
        
        image = datum['image'].swapaxes(0,1)
        keypoints = []
        for cname in self.class_labels:
            keypoints.append((datum[cname][1],datum[cname][0]))
                
        distorted = self.aug(
            image=image,
            keypoints=keypoints,
            class_labels=self.class_labels,
        )
        distorted_keypoints = distorted['keypoints']
        distorted_class_labels = distorted['class_labels']
        heatmaps = {}
        for cname in self.class_labels:
            if cname in distorted_class_labels:
                r, c = distorted_keypoints[distorted_class_labels.index(cname)]
                heatmaps[cname] = self.gaussian_heatmap(r,c,self.img_size)
            else:
                heatmaps[cname] = np.zeros(self.img_size,dtype=np.float32)


        return distorted['image'], heatmaps

    def gaussian_heatmap(self, r, c, shape, sigma=10):
        """
        Returns a heat map of a point
        Shape is expected to be (HEIGHT, WIDTH)
        [r,c] should be the point i.e. opposite of pygame notation.

        Parameters
        ----------
        r : int
            row of the point
        c : int
            column of the point
        shape : tuple of int
            (HEIGHT, WIDTH)

        Returns
        -------
        heatmap : np.array
            shape : (HEIGHT, WIDTH)
        """
        coordinates = np.stack(np.meshgrid(
            np.arange(shape[0],dtype=np.float32),
            np.arange(shape[1],dtype=np.float32),
            indexing='ij',
        ), axis=-1)
        keypoint = np.array([r,c]).reshape((1,1,2))
        heatmap = np.exp(-(np.sum((coordinates-keypoint)**2,axis=-1))/(2*sigma**2))

        return heatmap

class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.
    Only resizes the image
    """
    def __init__(self, data_dir, class_labels, img_size):
        """ 
        arguments
        ---------
        data_dir : str
            path to the data directory (which has pickled files)
        class_labels : list of str
            list of classes to extract data
        img_size : tuple
            Desired output image size
            IMPORTANT : (HEIGHT, WIDTH)
        """
        super().__init__(data_dir, class_labels, img_size)
        self.aug = A.Compose([
            A.Resize(img_size[0], img_size[1]),
        ],
        # Unify all points order to 'ij' format i.e. 'yx' format
        keypoint_params=A.KeypointParams(format='yx',label_fields=['class_labels'])
        )

def create_train_dataset(
        data_dir, 
        class_labels, 
        img_size, 
        batch_size, 
        buffer_size=1000,
        val_data=False):
    """
    Note: img_size = (HEIGHT,WIDTH)
    """
    autotune = tf.data.experimental.AUTOTUNE
    if val_data:
        generator = ValGenerator(
            data_dir,
            class_labels,
            img_size,
        )
    else:
        generator = AugGenerator(
            data_dir,
            class_labels,
            img_size,
        )
    output_dict = {}
    output_types = {}
    for cname in class_labels:
        output_dict[cname] = tf.TensorShape([img_size[0],img_size[1]])
        output_types[cname] = tf.float32
    

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.uint8, output_types),
        output_shapes=(
            tf.TensorShape([img_size[0],img_size[1],3]), 
            output_dict,
        ),
    )
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset


# def get_model(model_f, img_size):
#     """
#     To get model only and load weights.
#     """
#     # policy = mixed_precision.Policy('mixed_float16')
#     # mixed_precision.set_policy(policy)
#     inputs = keras.Input((img_size[0],img_size[1],3))
#     test_model = ClassifierModel(inputs, model_f)
#     test_model.compile(
#         optimizer='adam',
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=[
#             keras.metrics.SparseCategoricalAccuracy(),
#         ]
#     )
#     return test_model

class ValFigCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, logdir, class_labels):
        super().__init__()
        self.val_ds = val_ds
        self.filewriter = tf.summary.create_file_writer(logdir+'/val_image')
        self.class_labels = class_labels
        self.class_num = len(class_labels)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def val_result_fig(self):
        samples = self.val_ds.take(4).as_numpy_iterator()
        fig = plt.figure(figsize=(15,15))
        for i in range(4):
            sample = next(samples)
            sample_x = sample[0]
            predict = self.model(sample_x, training=False)
            for j, cname in enumerate(self.class_labels):
                sample_y = sample[1][cname][0]
                ax = fig.add_subplot(8,self.class_num,2*self.class_num*i+j+1)
                ax.imshow(sample_x[0], alpha=0.5)
                ax.imshow(sample_y,alpha=0.5)
                ax = fig.add_subplot(8,self.class_num,
                                     2*self.class_num*i+j+self.class_num+1)
                ax.imshow(sample_x[0], alpha=0.5)
                ax.imshow(predict[cname][0],alpha=0.5)

        return fig

    def on_epoch_end(self, epoch, logs=None):
        image = self.plot_to_image(self.val_result_fig())
        with self.filewriter.as_default():
            tf.summary.image('val prediction', image, step=epoch)

class MaxPointDistL2(keras.metrics.Metric):
    """MaxPointDistL2
    
    Pick a max value point from y_pred and y_true each, and calculate
    L2 distance between two points.
    
    """
    def __init__(self, name='max_point_distance', **kwargs):
        super().__init__(name=name,**kwargs)
        self.total = self.add_weight(name='mpd', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_max_pos = tf.cast(tf.unravel_index(tf.math.argmax(
            tf.reshape(y_true,(y_true.shape[0],-1)),
            axis=1
        ), y_true.shape[1:]),tf.float32)
        pred_max_pos = tf.cast(tf.unravel_index(tf.math.argmax(
            tf.reshape(y_pred,(y_pred.shape[0],-1)),
            axis=1
        ), y_pred.shape[1:]),tf.float32)
        l2_dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.squared_difference(
            true_max_pos, pred_max_pos),axis=0))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            l2_dist = tf.multiply(l2_dist, sample_weight)
        self.total.assign_add(tf.reduce_mean(l2_dist))
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    

def run_training(
        backbone_f,
        specific_fs, 
        lr_f, 
        name, 
        epochs,
        steps_per_epoch,
        batch_size,
        class_labels, 
        train_dir,
        val_dir,
        img_size,
        mixed_float = True,
        notebook = True,
        load_model_path = None,
        profile = False,
    ):
    """
    img_size:
        (HEIGHT, WIDTH)
    """

    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = keras.Input((img_size[0],img_size[1],3))
    mymodel = ChaserModel(inputs, backbone_f, specific_fs)
    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('loaded from : ' + load_model_path)
    loss = keras.losses.MeanSquaredError()
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[MaxPointDistL2(name='mpd'),]
    )

    logdir = 'logs/fit/' + name
    if profile:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch='7,9',
            update_freq='epoch'
        )
    else :
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch=0,
            update_freq='epoch'
        )
    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        save_weights_only=True,
        verbose=1
    )

    if notebook:
        tqdm_callback = TqdmNotebookCallback(metrics=['loss'],
                                            leave_inner=False)
    else:
        tqdm_callback = TqdmCallback()

    train_ds = create_train_dataset(
        train_dir,
        class_labels,
        img_size,
        batch_size,
        buffer_size=1000
    )
    val_ds = create_train_dataset(
        val_dir,
        class_labels,
        img_size,
        batch_size,
        buffer_size=100,
        val_data=True,
    )

    image_callback = ValFigCallback(val_ds, logdir, class_labels)

    mymodel.fit(
        x=train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        # steps_per_epoch=10,
        callbacks=[
            tensorboard_callback,
            lr_callback,
            save_callback,
            tqdm_callback,
            image_callback,
        ],
        verbose=0,
        validation_data=val_ds,
        validation_steps=100,
    )


    delta = time.time()-st
    hours, remain = divmod(delta, 3600)
    minutes, seconds = divmod(remain, 60)
    print(f'Took {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')


if __name__ == '__main__':
    import os
    data_dir = 'data/save'

    ds = create_train_dataset(data_dir, ['nose','tail'], (480,640),1,200)
    sample = ds.take(5).as_numpy_iterator()
    fig = plt.figure(figsize=(10,10))
    for i, s in enumerate(sample):
        ax = fig.add_subplot(5,3,3*i+1)
        img = s[0][0]
        ax.imshow(img)
        ax = fig.add_subplot(5,3,3*i+2)
        nose = s[1]['nose'][0]
        ax.imshow(img,alpha=0.5)
        ax.imshow(nose,alpha=0.5)
        ax = fig.add_subplot(5,3,3*i+3)
        tail = s[1]['tail'][0]
        ax.imshow(img,alpha=0.5)
        ax.imshow(tail,alpha=0.5)
    plt.show()