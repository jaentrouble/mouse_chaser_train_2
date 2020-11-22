import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

RPN_TRAIN_THRES = 0.5
BATCH_SIZE = 128
POSITIVE_RATIO = 0.5

class RPN(keras.Model):
    """RPN
    Gets a feature map and returns list of RoIs

    Note:
    Call method is not meant to be used when training.
    Use train_step directly.
    
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
        # Shape: (h,w,9,4)
        self._all_anchors = self.generate_anchors_pre(self.height, self.width)
        # Shape: (num_inside,4)
        self._inside_anchors = self.get_inside_anchors(self._all_anchors)

        self.cls_conv = layers.Conv2D(
            len(self.anchor_ratios)*len(self.anchor_scales),
            1,
        )
        self.reg_conv = layers.Conv2D(
            4*len(self.anchor_ratios)*len(self.anchor_scales),
            1,
        )

    def call(self, inputs, training=None):
        feature_map = self.inter_conv(inputs)

        cls_score = self.cls_conv(feature_map)
        bbox_reg = self.reg_conv(feature_map)
        except NotImplementedError

    def train_step(self, features, gt_boxes):
        """
        Parameters
        ----------
        features:
            Output of backbone models
        gt_boxes:
            Ground truth boxes
            Note: This model does not need tags.
        """
        with tf.GradientTape() as tape:
            feature_map = self.inter_conv(inputs)
            
            # Class loss
            cls_score = self.cls_conv(feature_map)
            



    def iou(self, bbox1, bbox2):
        """iou
        Calculate iou

        Parameters
        ----------
        bbox1, bbox2: tf.Tensor
            Broadcastable shape, and the last dimension should be 4 i.e. [...,4]

        Return
        ------
        iou: tf.Tensor
            Broadcasted shape, and the last axis is reduced.
        """
        # To prevent division by zero
        gamma_w = 1/self.width
        gamma_h = 1/self.height
        # x2,y2 must be bigger than x1,y1 all the time.
        # Do not add tf.abs because it may hide the problem
        S1 = tf.reduce_prod(
            bbox1[...,2:]-bbox1[...,0:2]+tf.constant([gamma_w,gamma_h]),
            axis=-1,
        )
        S2 = tf.reduce_prod(
            bbox2[...,2:]-bbox2[...,0:2]+tf.constant([gamma_w,gamma_h]),,
            axis=-1,
        )
        
        xA = tf.maximum(bbox1[...,0],bbox2[...,0])
        yA = tf.maximum(bbox1[...,1],bbox2[...,1])
        xB = tf.minimum(bbox1[...,2],bbox2[...,2])
        yB = tf.minimum(bbox1[...,3],bbox2[...,3])

        inter = tf.maximum((xB-xA+gamma_w),0) * tf.maximum((yB-yA+gamma_h),0)
        iou = inter/(S1 + S2 - inter)
        return iou

    def get_inside_anchors(self, anchors):
        """
        Only keep anchors inside the image
        """
        # Shape: (h,w,9)
        idx_inside = tf.where(
            (anchors[...,0] >= 0) &
            (anchors[...,1] >= 0) &
            (anchors[...,2] < 1) &
            (anchors[...,3] < 1)
        )
        # shape:
        inside_anchors = tf.gather_nd(
            anchors,
            idx_inside
        )
        return inside_anchors
    
    def generate_anchors_pre(self, height, width):
        x = tf.range(1, delta=1/width)
        y = tf.range(1, delta=1/height)
        # Default meshgrid indexing is 'xy'
        # Therefore, xx shape is [h,w]
        xx, yy = tf.meshgrid(x, y)
        
        # shape: (h,w,4)
        centers = tf.transpose(tf.stack([xx,yy,xx,yy]))
        # shape: (9, 4)
        anchor_sample = self.generate_anchors_set()
        # shape: (h,w,9,4)
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

    def anchor_target(self, gt_boxes):
        """Anchor_target
        Create target data
        """
        labels = tf.zeros((self._inside_anchors.shape[0],))
        # Shape: (n, 1, 4)
        i_anch_exp = tf.expand_dims(self._inside_anchors,1)
        # Shape: (1, k, 4)
        gt_boxes_exp = tf.expand_dims(gt_boxes,0)
        # Shape: (n, k)
        iou = self.iou(i_anch_exp, gt_boxes_exp)
        # Shape: (n,)
        argmax_iou = tf.argmax(iou, axis=1)
        max_iou = tf.reduce_max(iou, axis=1)

        # Shape: (k,)
        gt_max_iou = tf.reduce_max(iou, axis=0)
        # Shape: (k+a,) if multiple items with maximum value exists
        gt_argmax_iou = tf.where(iou == gt_max_iou)[:,0]

        labels = tf.tensor_scatter_nd_update(
            labels, 
            tf.expand_dims(gt_argmax_iou,-1),
            tf.ones(tf.shape(gt_argmax_iou)[0])
        )

        over_thres = tf.where(max_iou>=RPN_TRAIN_THRES)
        labels = tf.tensor_scatter_nd_update(
            labels, 
            over_thres,
            tf.ones(tf.shape(over_thres)[0])
        )
        
        # Subsample positive if too many
        max_p_num = tf.cast(POSITIVE_RATIO*BATCH_SIZE,tf.int64)
        p_idx = tf.where(labels==1)
        p_num = tf.shape(p_idx)[0]
        mixed_p_idx = tf.random.shuffle(p_idx)
        delta_p = p_num - max_p_num
        labels = tf.cond(
            delta_p > 0,
            lambda: tf.tensor_scatter_nd_update(
                labels, 
                mixed_p_idx[:delta_p],
                tf.ones(delta_p)*(-1),
            ),
            lambda: labels
        )

        # Subsample negative if too many
        max_n_num = BATCH_SIZE - tf.reduce_sum(tf.cast(labels==1,tf.float32))
        n_idx = tf.where(labels == 0)
        n_num = tf.shape(n_idx)[0]
        mixed_n_idx = tf.random.shuffle(n_idx)
        delta_n = n_num - max_n_num
        labels = tf.cond(
            delta_p > 0,
            lambda: tf.tensor_scatter_nd_update(
                labels,
                mixed_n_idx[:delta_n],
                tf.ones(delta_n)*(-1)
            ),
            lambda: labels
        )



    def get_config(self):
        config = super().get_config()
        config['intermediate_filters'] = self.intermediate_filters
        config['width'] = self.width
        config['height'] = self.height
        config['kernel_size'] = self.kernel_size
        config['stride'] = self.stride

        return config
