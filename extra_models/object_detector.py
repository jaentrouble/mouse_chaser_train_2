import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from extra_models import backbone_models

# TODO: debugging
import sys

RPN_TRAIN_THRES = 0.5
BATCH_SIZE = 128
POSITIVE_RATIO = 0.5

# NOTE: Here, 9 stands for anchor_set_num

class ObjectDetector(keras.Model):
    """ObjectDetector
    Gets an image and returns detected boundary boxes with classes

    Note:
    Call method is not meant to be used when training.
    Use train_step directly.
    
    RoI : (x1, y1, x2, y2), all normalized to [0,1]
        x: width
        y: height

    """
    def __init__(
        self, 
        backbone_layers,
        intermediate_filters,
        kernel_size,
        stride,
        image_size,
        anchor_ratios=[0.5,1.0,2.0], 
        anchor_scales=[0.2,0.4,0.7]
    ):
        """
        Arguments
        ---------
        backbone_layers: keras.Model
            A model that takes images and retuns features
        intermediate_filters: int
            filter number of the first conv layer
        kernel_size: int
            kernel size of the first conv layer
        stride: int
            stride of the conv layer
        image_size: tuple
            (WIDTH, HEIGHT) of the original input image
        anchor_ratios: list
            list of anchor shapes (width/height)
        anchor_scales: list
            list of anchor sizes
        """
        super().__init__()
        self.backbone_layers = backbone_layers
        self.intermediate_filters = intermediate_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.image_size = image_size
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.loss_tracker = keras.metrics.Mean(name='loss')

        self.inter_conv = layers.Conv2D(
            self.intermediate_filters,
            self.kernel_size,
            strides=self.stride,
        )
        
        self.anchor_set_num = len(self.anchor_ratios)*len(self.anchor_scales)

        self.cls_conv = layers.Conv2D(
            self.anchor_set_num,
            1,
        )
        self.reg_conv = layers.Conv2D(
            4*self.anchor_set_num,
            1,
        )

        

    def call(self, inputs, training=None):
        features = self.backbone_layers(inputs)
        
        feature_map = self.inter_conv(inputs)

        cls_score = self.cls_conv(feature_map)
        bbox_reg = self.reg_conv(feature_map)
        raise NotImplementedError

    def train_step(self, data):
        """
        Parameters
        ----------
        data: (image, gt_boxes, classes)
            image:
                Shape: (1,H,W,3)
            gt_boxes:
                Ground truth boxes
            classes:
                Ground truth classes of gt_boxes, in the same order
        """
        # Data comes in ((image, gt_boxes, classes),)
        image, gt_boxes, classes = data[0]
        gt_boxes = gt_boxes[0]

        with tf.GradientTape() as tape:
            features = self.backbone_layers(image, training=True)
            # Shape: (N,H,W,C)
            feature_map = self.inter_conv(features)

            # Shape: (h,w,9,4)
            f_height = feature_map.shape[1]
            f_width = feature_map.shape[2]
            all_anchors = self.generate_anchors_pre(f_height,f_width)
            # Shape: (num_inside,3), (num_inside,4)
            idx_inside, inside_anchors = \
                self.get_inside_anchors(all_anchors)
            rpn_labels, rpn_bbox_targets, rpn_bbox_mask = \
                self.anchor_target(idx_inside, inside_anchors, gt_boxes,
                                f_height, f_width)
            # Shape: (num_not_-1, 4), 4 for (1, height, width, 9)
            rpn_select = tf.where(tf.not_equal(
                rpn_labels,
                -1
            ))
                
            # Class loss
            # Shape: (1, height, width, 9), Batch should be 1
            cls_score = self.cls_conv(feature_map)
            # Shape: (num_not_-1,)
            rpn_selected_cls_score = tf.gather_nd(
                cls_score,
                rpn_select,
            )
            rpn_selected_labels = tf.gather_nd(
                rpn_labels,
                rpn_select,
            )
            rpn_cls_loss = tf.reduce_mean(tf.losses.binary_crossentropy(
                rpn_selected_labels,
                rpn_selected_cls_score,
                from_logits=True,
            ))

            # Reg loss
            bbox_pred = self.reg_conv(feature_map)
            bbox_pred = tf.reshape(bbox_pred,[
                tf.shape(bbox_pred)[0],
                tf.shape(bbox_pred)[1],
                tf.shape(bbox_pred)[2],
                self.anchor_set_num,
                4,
            ])
            rpn_bbox_loss = self.smooth_l1_loss(
                bbox_pred, rpn_bbox_targets, rpn_bbox_mask 
            )
            loss = rpn_cls_loss + rpn_bbox_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker]



    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_mask, sigma=1.0):
        """smooth_l1_loss
        Let x be target-pred
        if abs(x) < 1: 0.5*x**2
        else: abs(x) - 0.5

        Parameters
        ----------
        bbox_pred, bbox_targets: tf.Tensor
            Shape: (1, height, width, 9, 4)
            Order does not matter
        bbox_mask: tf.Tensor
            Shape: (1, height, width, 9, 1)
            Will be multiplied to loss
        sigma: float
            How steep L2 part is.
            i.e. y=sigma**2 * x
        """
        sigma_2 = sigma**2
        box_diff = bbox_pred - bbox_targets
        masked_box_diff = bbox_pred * bbox_mask
        abs_box_diff = tf.abs(masked_box_diff)
        smooth_sign = tf.cast(abs_box_diff<(1/sigma_2),tf.float32)
        box_loss = (abs_box_diff**2)*(sigma_2/2)*smooth_sign \
                   + (abs_box_diff - (0.5/sigma_2))*(1-smooth_sign)
        box_loss = tf.reduce_sum(box_loss) / tf.reduce_sum(bbox_mask)
        return box_loss



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
        gamma_w = 1/self.image_size[0]
        gamma_h = 1/self.image_size[1]
        print(f'bbox1 shape: {bbox1.shape}')
        print(f'bbox2 shape: {bbox2.shape}')
        # x2,y2 must be bigger than x1,y1 all the time.
        # Do not add tf.abs because it may hide the problem
        S1 = tf.reduce_prod(
            bbox1[...,2:]-bbox1[...,0:2]+tf.constant([gamma_w,gamma_h]),
            axis=-1,
        )
        print(f'S1shape : {S1.shape}')
        S2 = tf.reduce_prod(
            bbox2[...,2:]-bbox2[...,0:2]+tf.constant([gamma_w,gamma_h]),
            axis=-1,
        )
        
        xA = tf.maximum(bbox1[...,0],bbox2[...,0])
        print(bbox2[...,0].shape)
        yA = tf.maximum(bbox1[...,1],bbox2[...,1])
        xB = tf.minimum(bbox1[...,2],bbox2[...,2])
        yB = tf.minimum(bbox1[...,3],bbox2[...,3])
        print(f'xA shape:{xA.shape}')

        inter = tf.maximum((xB-xA+gamma_w),0) * tf.maximum((yB-yA+gamma_h),0)
        iou = inter/(S1 + S2 - inter)
        return iou

    def get_inside_anchors(self, anchors):
        """
        Only keep anchors inside the image
        """
        # Shape: (num_inside,3), 3 for (h,w,9)
        mask_inside = \
            (anchors[...,0] >= 0.0) &\
            (anchors[...,1] >= 0.0) &\
            (anchors[...,2] < 1.0) &\
            (anchors[...,3] < 1.0)
        idx_inside = tf.where(mask_inside)
        # Shape: (num_inside,4)
        inside_anchors = tf.gather_nd(
            anchors,
            idx_inside
        )
        print('-----here----')
        print(inside_anchors.shape)
        return idx_inside, inside_anchors
    
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

    def anchor_target(
        self, 
        idx_inside, 
        inside_anchors, 
        gt_boxes,
        f_height,
        f_width,
    ):
        """Anchor_target
        Create target data

        Parameters
        ----------
        idx_inside:
            Shape: (num_inside,3), 3 for (H,W,9)
            Indices of anchors that are completely inside the image
        inside_anchors:
            Shape: (num_inside,4)
        gt_boxes:
            Shape: (k, 4)
        f_height, f_width: int
            final feature shape
        """
        num_inside = tf.shape(inside_anchors)[0]
        # Shape: (num_inside,)
        labels = tf.fill([num_inside,],0.0)
        # Shape: (num_inside, 1, 4)
        i_anch_exp = tf.expand_dims(inside_anchors,1)
        print(i_anch_exp)
        # Shape: (1, k, 4)
        gt_boxes_exp = tf.expand_dims(gt_boxes,0)
        print(gt_boxes)
        # Shape: (num_inside, k)
        iou = self.iou(i_anch_exp, gt_boxes_exp)
        print(f'iou shape {iou.shape}')
        # Shape: (num_inside,)
        argmax_iou = tf.argmax(iou, axis=1)
        print(f'argmax_ioushape {argmax_iou.shape}')
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
        max_p_num = tf.cast(POSITIVE_RATIO*BATCH_SIZE,tf.int32)
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
        max_n_num = BATCH_SIZE - tf.reduce_sum(tf.cast(labels==1,tf.int32))
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
        
        gt_gathered = tf.gather_nd(
            gt_boxes,
            tf.expand_dims(argmax_iou,-1),
        )
        # Shape: (num_inside, 4)
        bbox_targets = self.bbox_delta_transform(
            inside_anchors, gt_gathered)
        
        # Only the positive ones have regression targets
        p_idx = tf.where(labels==1)
        p_num = tf.shape(p_idx)[0]
        # Shape: (num_inside, 1)
        bbox_mask = tf.scatter_nd(
            p_idx,
            tf.ones((p_num,1)),
            [num_inside,1],
        )

        # Shape: (height, width, 9)
        rpn_labels = tf.tensor_scatter_nd_update(
            tf.fill([f_height, f_width, self.anchor_set_num], -1.0),
            idx_inside,
            labels,
        )
        # Shape: (1, height, width, 9)
        rpn_labels = tf.expand_dims(rpn_labels, axis=0)

        rpn_bbox_targets = tf.tensor_scatter_nd_update(
            tf.zeros([
                f_height,
                f_width,
                self.anchor_set_num,
                4,
            ]),
            idx_inside,
            bbox_targets,
        )
        # Shape: (1, height, width, 9, 4)
        rpn_bbox_targets = tf.expand_dims(rpn_bbox_targets,axis=0)

        rpn_bbox_mask = tf.tensor_scatter_nd_update(
            tf.zeros([
                f_height,
                f_width,
                self.anchor_set_num,
                1,
            ]),
            idx_inside,
            bbox_mask,
        )
        # Shape: (1, height, width, 9, 1)
        rpn_bbox_mask = tf.expand_dims(rpn_bbox_mask, axis=0)


        return rpn_labels, rpn_bbox_targets, rpn_bbox_mask




    def bbox_delta_transform(self, an, gt):
        """
        Calculate distance between anchors and ground truth.
        This is the value that reg layer should predict.

        Parameters
        ----------
        an: tf.Tensor
            Anchors
        gt: tf.Tensor
            Ground truth
        Return
        ------
        targets: tf.Tensor
            last dimension: (dx, dy, dw, dh)
            dx, dy: normalized to the anchor's size
            dw, dh: log difference
        """
        g_width = 1/self.image_size[0]
        g_height = 1/self.image_size[1]
        an_widths = an[...,2] - an[...,0] + g_width
        an_heights = an[...,3] - an[...,1] + g_height
        an_ctr_x = an[...,0] + 0.5 * an_widths
        an_ctr_y = an[...,1] + 0.5 * an_heights

        gt_widths = gt[...,2] - gt[...,0] + g_width
        gt_heights = gt[...,3] - gt[...,1] + g_height
        gt_ctr_x = gt[...,0] + 0.5 * gt_widths
        gt_ctr_y = gt[...,1] + 0.5 * gt_heights

        dx = (gt_ctr_x - an_ctr_x) / an_widths
        dy = (gt_ctr_y - an_ctr_y) / an_heights
        dw = tf.math.log(gt_widths/an_widths)
        dh = tf.math.log(gt_heights/an_heights)

        target = tf.stack([dx,dy,dw,dh], axis=-1)
        return target

    def get_config(self):
        config = super().get_config()
        config['intermediate_filters'] = self.intermediate_filters
        config['kernel_size'] = self.kernel_size
        config['stride'] = self.stride
        config['image_size'] = self.image_size
        config['backbone_layers'] = self.backbone_layers
        config['anchor_ratios'] = self.anchor_ratios
        config['anchor_scales'] = self.anchor_scales

        return config
