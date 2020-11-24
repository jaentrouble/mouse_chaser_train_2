import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import time
from datetime import datetime
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
from extra_models.object_detector import ObjectDetector
from extra_models import backbone_models

# Min 100px, +- 30 px
# Minimum size of any side of a mouse bbox
MOUSE_MIN = 100
# Margin from nose/tail
MOUSE_MARGIN = 30

class AugGenerator():
    """An iterable generator that makes augmented mouserec data

    NOTE: 
        Every img is reshaped to img_size

    Returns
    -------
    image : np.array, dtype= np.float32
        Normalized to [0,1]
        shape : (HEIGHT, WIDTH, 3)
        color order : RGB
    gt_boxes: np.array, dtype= np.float32
        Normalized to [0,1]
        shape : (k, 4) where k is number of boxes
    gt_classes: np.array, dtype= np.int32
        Class index for gt_boxes
        Mouse is always labeled 0
        Classes will be labeled from 1 and in order of class_names
        Shape : (k, )

    """
    def __init__(self, data_dir, img_size, class_names=None, bbox_sizes=None):
        """
        Arguments
        ---------
        data_dir : str
            path to the data directory (which has pickled files)
        img_size : tuple
            Desired output image size
            IMPORTANT : (WIDTH, HEIGHT)
        class_names : list
            list of classes to extract data other than mouse
        bbox_sizes : list
            Will draw a box which is centered at the class's point,
            as the size defined.
            The order of class names and bbox_sizes should match.
            [(half_width, half_height)]
            i.e. x_min = ctr_x - half_width
        """
        self.class_names = ['mouse']
        if not(class_names is None):
            self.class_names.extend(class_names)
        self.bbox_sizes = bbox_sizes
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
            A.Resize(img_size[1], img_size[0]),
            # A.Cutout(8,img_size[0]//12,img_size[1]//12)
        ],
        # (x1, y1, x2, y2) format, all normalized
        bbox_params=A.BboxParams(format='albumentations', 
                            label_fields=['bbox_labels']),
        )

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.n)
        datum = self.raw_data[idx]
        
        # swap to shape (h,w,3)
        image = datum['image'].swapaxes(0,1)
        bboxes = []
        bbox_labels = []
        height, width = image.shape[:2]

        # Mouse box
        nose_x, nose_y = datum['nose']
        tail_x, tail_y = datum['tail']
        x_ctr = (nose_x + tail_x)/2
        y_ctr = (nose_y + tail_y)/2
        m_width = np.max([np.abs(nose_x-tail_x)+MOUSE_MARGIN*2, MOUSE_MIN])
        m_height = np.max([np.abs(nose_y-tail_y)+MOUSE_MARGIN*2, MOUSE_MIN])
        x_min = (x_ctr - m_width/2) / width
        x_max = (x_ctr + m_width/2) / width
        y_min = (y_ctr - m_height/2) / height
        y_max = (y_ctr + m_height/2) / height
        bboxes.append([x_min,y_min,x_max,y_max])
        bbox_labels.append('mouse')

        if len(self.class_names)>1:
            for cname, half_size in zip(self.class_names[1:], self.bbox_sizes):
                if (len(np.shape(datum[cname])) == 1) and \
                    len(datum[cname])>0:
                    # Only one point
                    ctr_x, ctr_y = datum[cname]
                    h_width, h_height = half_size
                    x_min = (ctr_x - h_width)/width
                    x_max = (ctr_x + h_width)/width
                    y_min = (ctr_y - h_height)/height
                    y_max = (ctr_y + h_height)/height
                    bboxes.append([x_min,y_min,x_max,y_max])
                    bbox_labels.append(cname)
                elif len(datum[cname])>=1:
                    # Multiple points
                    for ctr_x, ctr_y in datum[cname]:
                        h_width, h_height = half_size
                        x_min = (ctr_x - h_width)/width
                        x_max = (ctr_x + h_width)/width
                        y_min = (ctr_y - h_height)/height
                        y_max = (ctr_y + h_height)/height
                        bboxes.append([x_min,y_min,x_max,y_max])
                        bbox_labels.append(cname)
                    
        
        bboxes = np.clip(bboxes, 0, 1)

        distorted = self.aug(
            image=image,
            bboxes=bboxes,
            bbox_labels=bbox_labels,
        )
        t_box = np.array(distorted['bboxes'], dtype=np.float32)
        t_labels = np.array([
            self.class_names.index(cname) for cname in distorted['bbox_labels']
        ], dtype=np.float32)
        t_image = (distorted['image']/255).astype(np.float32)

        # Just in case there's no box left, retry
        if len(t_box) < 1:
            return self.__next__()

        return t_image, t_box, t_labels


class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.

    Only resizes the image
    """
    def __init__(self, data_dir, img_size, class_names=None, bbox_sizes=None):
        super().__init__(data_dir, img_size, class_names, bbox_sizes)
        self.aug = A.Compose([
            A.Resize(img_size[1], img_size[0]),
        ],
        # (x1, y1, x2, y2) format, all normalized
        bbox_params=A.BboxParams(format='albumentations', 
                            label_fields=['bbox_labels']),
        )

def create_train_dataset(
        data_dir, 
        img_size, 
        class_names=None,
        bbox_sizes=None,
        buffer_size=1000,
        val_data=False):
    """
    Note: img_size = (WIDTH,HEIGHT)
    Batch size is fixed to 1, because object detector model
    only takes one image per step
    """
    autotune = tf.data.experimental.AUTOTUNE
    if val_data:
        generator = ValGenerator(
            data_dir,
            img_size,
            class_names,
            bbox_sizes
        )
    else:
        generator = AugGenerator(
            data_dir,
            img_size,
            class_names,
            bbox_sizes
        )
    

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([img_size[1],img_size[0],3]), 
            tf.TensorShape([None, 4]),
            tf.TensorShape([None,])
        ),
    )
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(1)
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
    def __init__(self, val_ds, logdir):
        super().__init__()
        self.val_ds = val_ds
        self.filewriter = tf.summary.create_file_writer(logdir+'/val_image')
        self.colors=np.array([
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [0,1,1],
        ])

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
            image, gt_box, _ = sample
            # Shape: (n,4), (n,)
            boxes, probs, labels = self.model(image, training=True)
            test_image = image[0].copy()
            gt_image = image[0].copy()
            # Shape: (k,4)
            gt_box = gt_box[0]
            h,w = np.subtract(gt_image.shape[:2],1)
            for box, p, l in zip(boxes,probs, labels):
                color = self.colors[l] * p
                x1, y1, x2, y2 = np.multiply(box,[w,h,w,h,]).astype(np.int64)
                test_image[y1,x1:x2] = color
                test_image[y2,x1:x2] = color
                test_image[y1:y2,x1] = color
                test_image[y1:y2,x2] = color
            for box in gt_box:
                x1, y1, x2, y2 = np.multiply(box,[w,h,w,h,]).astype(np.int64)
                gt_image[y1,x1:x2] = [0,0,1]
                gt_image[y2,x1:x2] = [0,0,1]
                gt_image[y1:y2,x1] = [0,0,1]
                gt_image[y1:y2,x2] = [0,0,1]

            ax = fig.add_subplot(4,2,2*i+1)
            ax.imshow(gt_image)
            ax = fig.add_subplot(4,2,2*i+2)
            ax.imshow(test_image)

        return fig

    def on_epoch_end(self, epoch, logs=None):
        image = self.plot_to_image(self.val_result_fig())
        with self.filewriter.as_default():
            tf.summary.image('val prediction', image, step=epoch)


def run_training(
        backbone_f,
        lr_f, 
        name, 
        epochs,
        steps_per_epoch,
        # batch_size,
        intermediate_filters,
        kernel_size,
        stride,
        rfcn_window,
        anchor_ratios,
        anchor_scales,
        class_names, 
        bbox_sizes,
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
        (WIDTH, HEIGHT)
    """

    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    st = time.time()

    inputs = keras.Input((img_size[0],img_size[1],3))
    if class_names is None:
        num_classes = 1
    else:
        num_classes = len(class_names)
    mymodel = ObjectDetector(
        backbone_f,
        intermediate_filters,
        kernel_size,
        stride,
        img_size,
        num_classes,
        rfcn_window,
        anchor_ratios,
        anchor_scales
    )
    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('loaded from : ' + load_model_path)
    mymodel.compile(
        optimizer='adam',
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
        img_size,
        class_names,
        bbox_sizes,
        buffer_size=1000,
    )
    val_ds = create_train_dataset(
        val_dir,
        img_size,
        class_names,
        bbox_sizes,
        buffer_size=100,
        val_data=True,
    )

    image_callback = ValFigCallback(val_ds, logdir)

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
        # validation_data=val_ds,
        # validation_steps=100,
    )


    delta = time.time()-st
    hours, remain = divmod(delta, 3600)
    minutes, seconds = divmod(remain, 60)
    print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    print(f'Took {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')


if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    tf.config.set_visible_devices([],'GPU')
    img_size = (640,480)
    mymodel = ObjectDetector(
        'hr_5_3_8',
        256,
        16,
        8,
        img_size,
        3,
    )
    mymodel.load_weights('savedmodels/hr538_m_f2/91')
    mymodel.compile(optimizer='adam',run_eagerly=True)

    ds = create_train_dataset(
        'data/save',
        img_size,
        ['food'],
        [(20,20)],
    )
    from datetime import datetime
    now = datetime.now().strftime('%H_%M_%S')
    image_callback = ValFigCallback(ds, f'logs/fit/{now}')
    mymodel.fit(ds,steps_per_epoch=2,callbacks=[image_callback])
