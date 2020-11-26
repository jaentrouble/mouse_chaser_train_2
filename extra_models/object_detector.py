import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from extra_models import backbone_models

RPN_TRAIN_THRES = 0.5
BATCH_SIZE = 128
POSITIVE_RATIO = 0.5

NMS_TOP_N_RPN = 100
NMS_TOP_N_RFCN = 10
NMS_THRES_RPN = 0.5
NMS_THRES_RFCN = 0.1
NMS_SCORE_THRES_RPN = 0.4
NMS_SCORE_THRES_RFCN = 0.5
SOFT_SIGMA_RPN = 1.0 
SOFT_SIGMA_RFCN = 0.0

RPN_LOSS_GAMMA = 1.0
RFCN_LOSS_GAMMA = 1.0 # <-----------------Changed(rfcn_mf8)

SMOOTH_L1_SIGMA = 1.0
BG_TRAIN_RATIO = 0.75 # <----------------Name changed
FG_THRES = 0.5
BBOX_LOSS_GAMMA_RPN = 1.0 # <------------Changed(rfcn_mf8)
BBOX_LOSS_GAMMA_RFCN = 1.0 

ALIGN_RES = 10 # Crop and resize to (ALIGN_RES*k, ALIGN_RES*k)
OHEM_N = 100


# NOTE: Here, 9 stands for anchor_set_num

class ObjectDetector(keras.Model):
    """ObjectDetector
    Gets an image and returns detected boundary boxes with classes

    Note:
        This model only takes a single image at a time.

    Note:
        Call method is not meant to be used when training.
        Use train_step directly.
    
    RoI : (x1, y1, x2, y2), all normalized to [0,1]
        x: width
        y: height

    """
    def __init__(
        self, 
        backbone_f,
        intermediate_filters,
        kernel_size,
        stride,
        image_size,
        num_classes,
        rfcn_window,
        anchor_ratios, 
        anchor_scales,
    ):
        """
        Arguments
        ---------
        backbone_f: str
            A function to build a backbone model
        intermediate_filters: int
            filter number of the first conv layer
        kernel_size: int
            kernel size of the first conv layer
        stride: int
            stride of the conv layer
        image_size: tuple
            (WIDTH, HEIGHT) of the original input image
        num_classes: int
            Number of object classes
        rfcn_window: int
            R-FCN pooling window's size (k)
        anchor_ratios: list
            list of anchor shapes (width/height)
        anchor_scales: list
            list of anchor sizes
        """
        super().__init__()
        self.backbone_f = backbone_f
        self.intermediate_filters = intermediate_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.image_size = image_size
        self.num_classes = num_classes
        self.rfcn_window = rfcn_window
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.rpn_loss_tracker = keras.metrics.Mean(name='rpn_loss')
        self.rfcn_loss_tracker = keras.metrics.Mean(name='rfcn_loss')
        self.precision_metric = keras.metrics.SparseCategoricalAccuracy(name='prec')
        self.sensitivity_metric = keras.metrics.SparseCategoricalAccuracy(name='sens')

        image_w, image_h = image_size
        backbone_inputs = keras.Input((image_h,image_w,3))
        backbone_outputs = getattr(backbone_models,backbone_f)(backbone_inputs)
        self.backbone_model = keras.Model(inputs=backbone_inputs,
                                        outputs=backbone_outputs,)

        self.rpn_inter_conv = layers.Conv2D(
            self.intermediate_filters,
            self.kernel_size,
            strides=self.stride,
        )
        # Dummy to determine layer's shape
        dummy_rpn_inter_output = self.rpn_inter_conv(backbone_outputs)

        self.f_height, self.f_width = \
            dummy_rpn_inter_output.shape[1:3]
        # Shape: (h,w,9,4)
        self._all_anchors = self.generate_anchors_pre(self.f_height, self.f_width)
        # Shape: (num_inside,3), (num_inside,4)
        self._idx_inside, self._inside_anchors = \
            self.get_inside_anchors(self._all_anchors)
        
        self.anchor_set_num = len(self.anchor_ratios)*len(self.anchor_scales)

        self.rpn_cls_conv = layers.Conv2D(
            self.anchor_set_num,
            1,
            dtype=tf.float32,
        )
        self.rpn_reg_conv = layers.Conv2D(
            4*self.anchor_set_num,
            1,
            dtype=tf.float32,
        )

        self.rfcn_cls_conv = layers.Conv2D(
            (self.rfcn_window**2)*(self.num_classes+1),
            3,
            padding='same',
            dtype=tf.float32,
        )
        self.rfcn_reg_conv = layers.Conv2D(
            (self.rfcn_window**2)*4,
            3,
            padding='same',
            dtype=tf.float32,
        )
        # Dummy to determine layer's shape
        dummy_rpn_cls_output = self.rpn_cls_conv(dummy_rpn_inter_output)
        dummy_rpn_reg_output = self.rpn_reg_conv(dummy_rpn_inter_output)
        dummy_rfcn_cls_output= self.rfcn_cls_conv(backbone_outputs)
        dummy_rfcn_reg_conv  = self.rfcn_reg_conv(backbone_outputs)


    def call(self, inputs, training=False):
        #-------DEBUG
        # gt = inputs[0]
        #---------------

        features = self.backbone_model(inputs, training=training)
        rpn_features = self.rpn_inter_conv(features)
        cls_score = self.rpn_cls_conv(rpn_features)
        bbox_reg = self.rpn_reg_conv(rpn_features)
        #-----------DEBUG
        # bbox_reg = tf.zeros_like(bbox_reg)
        #---------------------------------

        rois, rpn_soft_probs = self.rpn_proposal(cls_score, bbox_reg)

        #------------DEBUG
        # rpn_labels, boxes, _ = self.anchor_target(gt)
        # rpn_select = tf.where(tf.not_equal(
        #     rpn_labels,
        #     -1
        # ))
        # boxes = tf.gather_nd(
        #     tf.expand_dims(self._all_anchors,0),
        #     rpn_select,
        # )
        # soft_probs = tf.gather_nd(
        #     rpn_labels,
        #     rpn_select,
        # )
        #----------------------------------------------------
        rfcn_cls_features = self.rfcn_cls_conv(features)
        rfcn_cls_score = self.rfcn_cls_scores(rfcn_cls_features, rois)
        rfcn_reg_features = self.rfcn_reg_conv(features)
        rfcn_bbox_pred = self.rfcn_bbox_reg(rfcn_reg_features, rois)
        boxes, soft_probs, labels = self.rfcn_proposal(
            rfcn_cls_score, rfcn_bbox_pred, rois,
        )

        # return boxes, soft_probs, labels
        #---------------DEBUG
        return rois, rpn_soft_probs, boxes, soft_probs, labels

    def train_step(self, data):
        """
        Parameters
        ----------
        data: (image, gt_boxes, gt_labels)
            image:
                Shape: (1,H,W,3)
            gt_boxes:
                Ground truth boxes
            gt_labels:
                Ground truth labels of gt_boxes, in the same order
        """
        image, gt_boxes, gt_labels = data
        gt_boxes = gt_boxes[0]
        gt_labels = gt_labels[0]

        with tf.GradientTape() as tape:
            features = self.backbone_model(image, training=True)
            # Shape: (1,H,W,C)
            rpn_features = self.rpn_inter_conv(features)

            rpn_labels, rpn_bbox_targets, rpn_bbox_mask = \
                self.anchor_target(gt_boxes)
            # Shape: (num_not_-1, 4), 4 for (1, height, width, 9)
            rpn_select = tf.where(tf.not_equal(
                rpn_labels,
                -1
            ))
                
            # RPN Class loss
            # Shape: (1, height, width, 9), Batch should be 1
            rpn_cls_score = self.rpn_cls_conv(rpn_features)
            # Shape: (num_not_-1,)
            rpn_selected_cls_score = tf.gather_nd(
                rpn_cls_score,
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

            # RPN Reg loss
            rpn_bbox_pred = self.rpn_reg_conv(rpn_features)
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred,[
                tf.shape(rpn_bbox_pred)[0],
                tf.shape(rpn_bbox_pred)[1],
                tf.shape(rpn_bbox_pred)[2],
                self.anchor_set_num,
                4,
            ])
            rpn_bbox_loss = self.smooth_l1_loss(
                rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_mask 
            )
            rpn_bbox_loss = tf.reduce_sum(rpn_bbox_loss)
            rpn_loss = rpn_cls_loss + BBOX_LOSS_GAMMA_RPN*rpn_bbox_loss

            # RoI Proposal
            rois, soft_probs = self.rpn_proposal(rpn_cls_score, rpn_bbox_pred)
            rfcn_labels, rfcn_bbox_targets, rfcn_bbox_mask =\
                self.proposal_target(rois, gt_boxes, gt_labels)

            # R_FCN Class loss
            # Shape: (N,)
            rfcn_cls_mask = self.rfcn_limit_bg(rfcn_bbox_mask)
            rfcn_cls_features = self.rfcn_cls_conv(features)

            # Shape: (N, cls+1)
            rfcn_cls_score = self.rfcn_cls_scores(
                rfcn_cls_features,
                rois
            )
            # Shape: (N,)
            rfcn_all_cls_loss = tf.losses.sparse_categorical_crossentropy(
                rfcn_labels,
                rfcn_cls_score,
                from_logits=True,
            )
            masked_rfcn_cls_loss = rfcn_all_cls_loss * rfcn_cls_mask
            sorted_rfcn_cls_loss = tf.sort(
                rfcn_all_cls_loss,
                direction='DESCENDING'
            )
            rfcn_cls_loss = tf.reduce_mean(
                sorted_rfcn_cls_loss[:OHEM_N]
            )

            # R_FCN Reg loss
            rfcn_reg_features = self.rfcn_reg_conv(features)
            rfcn_bbox_pred = self.rfcn_bbox_reg(
                rfcn_reg_features,
                rois
            )
            rfcn_bbox_mask = rfcn_bbox_mask[...,tf.newaxis]
            rfcn_all_bbox_loss = self.smooth_l1_loss(
                rfcn_bbox_pred, rfcn_bbox_targets, rfcn_bbox_mask
            )

            sorted_rfcn_bbox_loss = tf.sort(
                rfcn_all_bbox_loss,
                direction='DESCENDING'
            )
            rfcn_bbox_loss = tf.reduce_sum(
                sorted_rfcn_bbox_loss[:OHEM_N]
            )

            rfcn_loss = rfcn_cls_loss + BBOX_LOSS_GAMMA_RFCN*rfcn_bbox_loss

            loss = RPN_LOSS_GAMMA * rpn_loss + RFCN_LOSS_GAMMA * rfcn_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.rpn_loss_tracker.update_state(rpn_loss)
        self.rfcn_loss_tracker.update_state(rfcn_loss)
        # Only count non-backgrounds
        positive_idx = tf.where(
            tf.argmax(rfcn_cls_score,axis=-1)!=self.num_classes
        )
        pos_labels = tf.gather(rfcn_labels, positive_idx)
        pos_scores = tf.gather(rfcn_cls_score, positive_idx)
        self.precision_metric.update_state(pos_labels, pos_scores)

        real_pos_idx = tf.where(
            rfcn_labels < self.num_classes
        )
        real_pos_labels = tf.gather(rfcn_labels, real_pos_idx)
        real_pos_scores = tf.gather(rfcn_cls_score, real_pos_idx)
        self.sensitivity_metric.update_state(real_pos_labels, real_pos_scores)
        
        return {'loss': self.loss_tracker.result(),
                'rpn_loss': self.rpn_loss_tracker.result(),
                'rfcn_loss': self.rfcn_loss_tracker.result(),
                'prec': self.precision_metric.result(),
                'sens': self.sensitivity_metric.result(),}
    
    @property
    def metrics(self):
        return [self.loss_tracker, 
                self.rpn_loss_tracker,
                self.rfcn_loss_tracker,
                self.precision_metric,
                self.sensitivity_metric]

    def rfcn_cls_scores(self, features, rois):
        """rfcn_cls_pooling
        Average pool and retun class scores (in logit)

        Parameters
        ----------
        features:
            Feature map
            Shape: (1,h,w,(cls+1)*(k**2))
        rois:
            Shape: (N,4)

        Returns
        -------
        scores:
            Shape: (N,num_cls+1), 1 for background
        """
        f_height = tf.shape(features)[1]
        f_width = tf.shape(features)[2]
        k = self.rfcn_window
        n_cls = self.num_classes
        n_rois = tf.shape(rois)[0]

        # Shape: (N,k,k,(cls+1)*k*k)
        pooled = self.roi_align(features, rois, k)
        # Shape: (N,k*k,(cls+1)*k*k)
        flat_pooled = tf.reshape(pooled, [-1,k**2,(n_cls+1)*k*k])

        axis1 = tf.tile(tf.range(k**2),[n_cls+1])
        axis2 = tf.range(k*k*(n_cls+1))
        # Shape: (k*k*(cls+1),2)
        pool_idx = tf.stack([axis1,axis2],axis=1)
        # Shape: (N,k*k*(cls+1),2)
        pool_idx = tf.tile(pool_idx[tf.newaxis,...],[n_rois,1,1])

        # Shape: (N,k*k*(cls+1))
        pooled = tf.gather_nd(
            flat_pooled,
            pool_idx,
            batch_dims=1,
        )
        pooled_reshaped = tf.reshape(pooled,[n_rois,(n_cls+1),k*k])
        # Shape: (N,cls+1)
        scores = tf.reduce_mean(pooled_reshaped, axis=-1)
        return scores

    def rfcn_bbox_reg(self, features, rois):
        """rfcn_bbox_reg
        Average pool and return bbox_reg

        Parameters
        ----------
        features:
            Feature map
            Shape: (1,h,w,4*(k**2))
        rois:
            Shape: (N,4)

        Returns
        -------
        bbox_reg:
            Shape: (N,4)
        """
        f_height = tf.shape(features)[1]
        f_width = tf.shape(features)[2]
        k = self.rfcn_window
        n_rois = tf.shape(rois)[0]

        # Shape: (N,k,k,4*k*k)
        pooled = self.roi_align(features, rois, k)
        # Shape: (N,k*k,4*k*k)
        flat_pooled = tf.reshape(pooled, [-1,k**2,4*k*k])

        axis1 = tf.tile(tf.range(k**2),[4])
        axis2 = tf.range(k*k*4)
        # Shape: (k*k*4,2)
        pool_idx = tf.stack([axis1,axis2],axis=1)
        # Shape: (N,k*k*4,2)
        pool_idx = tf.tile(pool_idx[tf.newaxis,...],[n_rois,1,1])

        # Shape: (N,k*k*4)
        pooled = tf.gather_nd(
            flat_pooled,
            pool_idx,
            batch_dims=1,
        )
        pooled_reshaped = tf.reshape(pooled,[n_rois,4,k*k])
        # Shape: (N,4)
        bbox_reg = tf.reduce_mean(pooled_reshaped, axis=-1)
        return bbox_reg

    def rfcn_limit_bg(self, bbox_mask):
        """rfcn_limit_bg
        Counts background example numbers, and if there are too many bg,
        (i.e. more than BG_TRAIN_RATIO) drop random bg.

        Parameter
        ---------
        bbox_mask:
            True (or casted to True) if foreground, 
            False (or casted to False) if background
            Shape: (N,)

        Return
        ------
        loss_mask:
            Mask where dropped bgs are 0
            Shape: (N,)
        """
        # Shape: (p_num,)
        bool_mask = tf.cast(bbox_mask,tf.bool)
        fg_indices = tf.where(bool_mask)
        bg_indices = tf.where(tf.logical_not(bool_mask))
        total_num = tf.shape(bbox_mask[0])
        fg_num = tf.shape(fg_indices)[0]
        bg_num = tf.shape(bg_indices)[0]
        max_bg_num = tf.cast(
                tf.cast(fg_num,tf.float32)*BG_TRAIN_RATIO/(1-BG_TRAIN_RATIO),
                        tf.int32)

        loss_mask = tf.fill([total_num,],1.0)
        mixed_bg_idx = tf.random.shuffle(bg_indices)
        delta_bg = bg_num - max_bg_num
        loss_mask = tf.cond(
            delta_bg > 0,
            lambda: tf.tensor_scatter_update(
                loss_mask,
                mixed_bg_idx[:delta_bg],
                tf.fill([delta_bg],0.0)
            ),
            lambda: loss_mask
        )
        

        return loss_mask


    def roi_align(self, features, rois, k):
        """roi_align
        Crop features in the shape of RoI (expects it to be normalized)
        and average pools to the bin size

        Parameters
        ----------
        features:
            Feature map
            Shape: (1, h, w, c)
        rois:
            Shape: (N, 4), in order of (x1, y1, x2, y2)
        k: int
            Output size will be (k, k)
        
        Return
        ------
        pooled:
            Shape: (N,k,k,c)
        """
        cropped = tf.image.crop_and_resize(
            features,
            rois[...,::-1],
            tf.fill([tf.shape(rois)[0]],0),
            [ALIGN_RES*k, ALIGN_RES*k],
        )
        pooled = tf.nn.avg_pool2d(cropped,ALIGN_RES,ALIGN_RES,'SAME')
        return pooled
        

    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_mask, 
                       sigma=SMOOTH_L1_SIGMA):
        """smooth_l1_loss
        Let x be target-pred
        if abs(x) < 1: 0.5*x**2
        else: abs(x) - 0.5

        Reduces final axis

        Parameters
        ----------
        bbox_pred, bbox_targets: tf.Tensor
            Shape: (..., 4)
            Order does not matter
        bbox_mask: tf.Tensor
            Shape: (..., 1)
            Will be multiplied to loss
        sigma: float
            How steep L2 part is.
            i.e. y=(sigma**2) * x until y < 1
        """
        sigma_2 = sigma**2
        box_diff = bbox_pred - bbox_targets
        masked_box_diff = bbox_pred * bbox_mask
        abs_box_diff = tf.abs(masked_box_diff)
        smooth_sign = tf.cast(abs_box_diff<(1/sigma_2),tf.float32)
        box_loss = (abs_box_diff**2)*(sigma_2/2)*smooth_sign \
                   + (abs_box_diff - (0.5/sigma_2))*(1-smooth_sign)
        box_loss = tf.math.divide_no_nan(
            tf.reduce_sum(box_loss,axis=-1),tf.reduce_sum(bbox_mask)
        )
        return box_loss

    def rpn_proposal(self, rpn_cls_score, rpn_bbox_pred):
        """
        Parameters
        ----------
        rpn_cls_score: tf.Tensor
            Output of cls layer
            Expects logit; sigmoid is done here
            Shape: (h,w,9)
        rpn_bbox_pred: tf.Tensor
            Output of reg layer
            Will be flattened anyway, so shape does not matter
            Shape: (h,w,9*4)

        Returns
        -------
        boxes: tf.Tensor
            Proposed maximum N boxes
            Shape: (M, 4) where M<=N
        soft_probs: tf.Tensor
            Softened scores
            Shape: (M,)
        """
        scores = tf.reshape(rpn_cls_score, (-1,))
        probs = tf.math.sigmoid(scores)
        deltas = tf.reshape(rpn_bbox_pred,(-1,4))
        flattened_anchors = tf.reshape(self._all_anchors,(-1,4))

        proposals = self.bbox_delta_inv(flattened_anchors, deltas)
        proposals = tf.clip_by_value(proposals, 0, 1)

        indices, soft_probs = tf.image.non_max_suppression_with_scores(
            proposals,
            probs,
            NMS_TOP_N_RPN,
            iou_threshold=NMS_THRES_RPN,
            soft_nms_sigma=SOFT_SIGMA_RPN,
            score_threshold=NMS_SCORE_THRES_RPN,
        )

        boxes = tf.gather(proposals, indices)
        # boxes = proposals
        # soft_probs = probs
        return boxes, soft_probs

    def rfcn_proposal(self, rfcn_cls_score, rfcn_bbox_pred, rois):
        """
        Parameters
        ----------
        rfcn_cls_score: tf.Tensor
            Expects logit; sigmoid is done here
            Shape: (n,num_cls+1)
        rfcn_bbox_pred: tf.Tensor
            Shape: (n,4)

        Returns
        -------
        boxes: tf.Tensor
            Proposed maximum N boxes
            Shape: (M, 4) where M<=N
        soft_probs: tf.Tensor
            Softened scores
            Shape: (M,)
        labels:
            Predicted labels per boxes
        """
        # Shape: (n,cls+1)
        probs = tf.math.sigmoid(rfcn_cls_score)

        # probs to use for nms
        # Shape: (n,)
        max_probs = tf.reduce_max(probs, axis=-1,keepdims=True)
        max_mask = tf.cast(probs==max_probs,tf.float32)
        # Leave only maximum probs per block
        clean_probs = probs*max_mask

        box_labels = tf.argmax(probs,axis=-1)

        deltas = rfcn_bbox_pred
        proposals = self.bbox_delta_inv(rois, deltas)
        proposals = tf.clip_by_value(proposals,0,1)

        box_outputs = []
        soft_probs_outputs = []
        class_outputs = []
        
        # Do not collect backgrounds
        for i in range(self.num_classes):
            indices, soft_probs = tf.image.non_max_suppression_with_scores(
                proposals,
                clean_probs[:,i],
                NMS_TOP_N_RFCN,
                iou_threshold=NMS_THRES_RFCN,
                soft_nms_sigma=SOFT_SIGMA_RFCN,
                score_threshold=NMS_SCORE_THRES_RFCN,
            )
            box_outputs.append(tf.gather(proposals, indices))
            soft_probs_outputs.append(soft_probs)
            class_outputs.append(tf.gather(box_labels,indices))
        
        final_boxes = tf.concat(box_outputs,axis=0)
        final_soft_probs = tf.concat(soft_probs_outputs,axis=0)
        final_labels = tf.concat(class_outputs,axis=0)

        return final_boxes, final_soft_probs, final_labels


    def proposal_target(self, rois, gt_boxes, gt_labels):
        """
        Assign target gt_box and labels to rois
        
        Parameters
        ----------
        rois:
            Shape: (M, 4)
        gt_boxes:
            Shape: (k, 4)
        gt_labels:
            Shape: (k,)

        Returns
        -------
        rfcn_labels:
            Shape: (M,)
        rfcn_bbox_targets:
            Shape: (M,4)
        rfcn_bbox_mask:
            Shape: (M,)
        """

        # Sample rois
        # Shape: (M,k)
        iou = self.iou(rois[:,tf.newaxis,:],gt_boxes)
        # Shape: (M,)
        argmax_iou = tf.argmax(iou, axis=1)
        # Shape: (M,)
        max_iou = tf.reduce_max(iou, axis=1)
        # Shape: (M,)
        cls_labels = tf.gather(
            gt_labels, 
            argmax_iou,
        )

        # Shape: (bg_num,1)
        bg_idx = tf.where(max_iou<FG_THRES)
        # Assign background the last label
        rfcn_labels = tf.tensor_scatter_nd_update(
            cls_labels,
            bg_idx,
            tf.fill([tf.shape(bg_idx)[0]],1)*self.num_classes,
        )

        # Shape: (M, 4)
        gt_targets = tf.gather(
            gt_boxes,
            argmax_iou,
        )
        rfcn_bbox_targets = self.bbox_delta_transform(
            rois,
            gt_targets,
        )
        rfcn_bbox_mask = tf.cast(rfcn_labels<self.num_classes, tf.float32)

        return rfcn_labels, rfcn_bbox_targets, rfcn_bbox_mask


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
        # x2,y2 must be bigger than x1,y1 all the time.
        # Do not add tf.abs because it may hide the problem
        S1 = tf.reduce_prod(
            bbox1[...,2:]-bbox1[...,0:2]+tf.constant([gamma_w,gamma_h]),
            axis=-1,
        )
        S2 = tf.reduce_prod(
            bbox2[...,2:]-bbox2[...,0:2]+tf.constant([gamma_w,gamma_h]),
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
        mask_inside = \
            (anchors[...,0] >= 0.0) &\
            (anchors[...,1] >= 0.0) &\
            (anchors[...,2] < 1.0) &\
            (anchors[...,3] < 1.0)
        idx_inside = tf.where(mask_inside)
        inside_anchors = tf.gather_nd(
            anchors,
            idx_inside
        )
        return idx_inside, inside_anchors
    
    def generate_anchors_pre(self, height, width):
        x = tf.range(1, delta=1/width)
        y = tf.range(1, delta=1/height)
        # Default meshgrid indexing is 'xy'
        # Therefore, xx shape is [h,w]
        xx, yy = tf.meshgrid(x, y)
        
        # shape: (h,w,4)
        centers = tf.stack([xx,yy,xx,yy],axis=-1)
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

    def anchor_target(self, gt_boxes,):
        """Anchor_target
        Create target data

        Parameters
        ----------
        gt_boxes:
            Shape: (k, 4)
        """
        num_inside = tf.shape(self._inside_anchors)[0]
        # Shape: (num_inside,)
        labels = tf.fill([num_inside,],0.0)
        # Shape: (num_inside, 1, 4)
        i_anch_exp = tf.expand_dims(self._inside_anchors,1)
        # Shape: (1, k, 4)
        gt_boxes_exp = tf.expand_dims(gt_boxes,0)
        # Shape: (num_inside, k)
        iou = self.iou(i_anch_exp, gt_boxes_exp)
        # Shape: (num_inside,)
        argmax_iou = tf.argmax(iou, axis=1)
        max_iou = tf.reduce_max(iou, axis=1)

        # Shape: (k,)
        gt_max_iou = tf.reduce_max(iou, axis=0)
        # Shape: (k+a,) if multiple items with maximum value exists
        gt_argmax_iou = tf.where(iou == gt_max_iou)[:,0]

        labels = tf.tensor_scatter_nd_update(
            labels, 
            tf.expand_dims(gt_argmax_iou,-1),
            tf.fill([tf.shape(gt_argmax_iou)[0]],1.0)
        )

        over_thres = tf.where(max_iou>=RPN_TRAIN_THRES)
        labels = tf.tensor_scatter_nd_update(
            labels, 
            over_thres,
            tf.fill([tf.shape(over_thres)[0]],1.0)
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
                tf.fill([delta_p,],-1.0),
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
            delta_n > 0,
            lambda: tf.tensor_scatter_nd_update(
                labels,
                mixed_n_idx[:delta_n],
                tf.fill([delta_n],-1.0)
            ),
            lambda: labels
        )
        
        gt_gathered = tf.gather_nd(
            gt_boxes,
            tf.expand_dims(argmax_iou,-1),
        )
        # Shape: (num_inside, 4)
        bbox_targets = self.bbox_delta_transform(
            self._inside_anchors, gt_gathered)
        
        # Only the positive ones have regression targets
        p_idx = tf.where(labels==1)
        p_num = tf.shape(p_idx)[0]
        # Shape: (num_inside, 1)
        bbox_mask = tf.scatter_nd(
            p_idx,
            tf.fill([p_num,1],1),
            [num_inside,1],
        )

        # Shape: (height, width, 9)
        rpn_labels = tf.tensor_scatter_nd_update(
            tf.fill([self.f_height, self.f_width, self.anchor_set_num], -1.0),
            self._idx_inside,
            labels,
        )
        # Shape: (1, height, width, 9)
        rpn_labels = tf.expand_dims(rpn_labels, axis=0)

        rpn_bbox_targets = tf.tensor_scatter_nd_update(
            tf.zeros([
                self.f_height,
                self.f_width,
                self.anchor_set_num,
                4,
            ]),
            self._idx_inside,
            bbox_targets,
        )
        # Shape: (1, height, width, 9, 4)
        rpn_bbox_targets = tf.expand_dims(rpn_bbox_targets,axis=0)

        rpn_bbox_mask = tf.tensor_scatter_nd_update(
            tf.zeros([
                self.f_height,
                self.f_width,
                self.anchor_set_num,
                1,
            ]),
            self._idx_inside,
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
    
    def bbox_delta_inv(self, boxes, deltas):
        """
        Inverse function of bbox_delta_transform

        Parameters
        ----------
        boxes: tf.Tensor
            Anchors (x1, y1, x2, y2)
        deltas:
            Output of reg layer (dx, dy, dw, dh)

        Return
        ------
        pred_boxes: tf.Tensor
            (x1, y1, x2, y2)
        """
        g_width = 1/self.image_size[0]
        g_height = 1/self.image_size[1]
        widths = boxes[...,2] - boxes[...,0] + g_width
        heights = boxes[...,3] - boxes[...,1] + g_height
        ctr_x = (boxes[...,0] + boxes[...,2])/2
        ctr_y = (boxes[...,1] + boxes[...,3])/2

        dx = deltas[...,0]
        dy = deltas[...,1]
        dw = deltas[...,2]
        dh = deltas[...,3]

        pred_ctr_x = (dx * widths) + ctr_x
        pred_ctr_y = (dy * heights) + ctr_y
        pred_w = tf.exp(dw) * widths
        pred_h = tf.exp(dh) * heights

        pred_x1 = pred_ctr_x - (pred_w * 0.5)
        pred_y1 = pred_ctr_y - (pred_h * 0.5)
        pred_x2 = pred_ctr_x + (pred_w * 0.5)
        pred_y2 = pred_ctr_y + (pred_h * 0.5)
        pred_boxes = tf.stack([
            pred_x1,
            pred_y1,
            pred_x2,
            pred_y2,
        ], axis=-1)
        return pred_boxes


    def get_config(self):
        config = super().get_config()
        config['intermediate_filters'] = self.intermediate_filters
        config['kernel_size'] = self.kernel_size
        config['stride'] = self.stride
        config['image_size'] = self.image_size
        config['backbone_f'] = self.backbone_f
        config['anchor_ratios'] = self.anchor_ratios
        config['anchor_scales'] = self.anchor_scales
        config['num_classes'] = self.num_classes
        config['rfcn_window'] = self.rfcn_window

        return config
