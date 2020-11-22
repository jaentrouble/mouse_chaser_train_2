import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

class RPN(keras.Model):
    """RPN
    Gets a feature map and returns list of RoIs
    
    RoI : (x1, y1, x2, y2), all normalized to [0,1]
        x: width
        y: height

    """
    def __init__(
        self, 
        intermediate_filters,
        kernel_size,
        stride,
        anchor_ratios=[0.5,1.0,2.0], 
        anchor_scales=[0.2,0.4,0.7]
    ):
        """
        Arguments
        ---------
        intermediate_filters:
            filter number of the first conv layer
        kernel_size: int
            kernel size of the first conv layer
        stride: int
            stride of the conv layer
        anchor_ratios: list
            list of anchor shapes (width/height)
        anchor_scales: list
            list of anchor sizes
        """
        self.intermediate_filters = intermediate_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales

    def build(self, input_shape):
        anchor_num = len(self.anchor_scales) * len(self.anchor_scales)
        self.inter_conv = layers.Conv2D(
            self.intermediate_filters,
            self.kernel_size,
            strides=self.stride,
        )
        # input is in n*h*w*c order
        self.width = tf.ceil((input_shape[-2]-self.kernel_size+1)/self.stride)
        self.height = tf.ceil((input_shape[-3]-self.kernel_size+1)/self.stride)

        self.cls_conv = layers.Conv2D(
            len(self.anchor_ratios)*len(self.anchor_scales),
            1,
        )
        self.reg_conv = layers.Conv2D(
            4*len(self.anchor_ratios)*len(self.anchor_scales),
            1,
        )

    
    def generate_anchors_pre(self, height, width):
        x = tf.range(width, delta=1/width)
        y = tf.range(height, delta=1/height)
        xx, yy = tf.meshgrid(x, y)
        
        # shape: (w,h,4)
        centers = tf.transpose(tf.stack([xx,yy,xx,yy]))
        # shape: (9, 4)
        anchor_sample = self.generate_anchors_set()
        # shape: (w,h,9,4)
        anchors_pre = tf.expand_dims(centers, axis=2) + anchor_sample

        return anchors_pre

    def generate_anchors_set(self):
        """
        Create a set of anchors.
        [-dx, -dy, dx, dy] where dx, dy are half of width, height each

        Return
        ------
        anchors: tf.Tensor
            shape : (len(anchor_ratios)*len(anchor_scales), 4)
        """
        ratio_sqrt = tf.sqrt(tf.expand_dims(self.anchor_ratios,axis=1))
        ws = tf.reshape(self.anchor_scales*ratio_sqrt,(-1,))
        hs = tf.reshape(self.anchor_scales/ratio_sqrt,(-1,))
        anchors = tf.transpose(tf.stack([ws/2,hs/2,ws/2,hs/2]))
        anchors = anchors * tf.constant([-1,-1,1,1],dtype=tf.float32)

        return anchors

    def get_config(self):
        config = super().get_config()
        config['intermediate_filters'] = self.intermediate_filters
        config['width'] = self.width
        config['height'] = self.height
        config['kernel_size'] = self.kernel_size
        config['stride'] = self.stride

        return config
