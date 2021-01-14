from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mxnet as mx
import cv2
import insightface
from insightface.utils import face_align

def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, prefix, epoch, layer):
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, ctx_id, model_prefix, model_epoch, use_large_detector=True, detector_type='retina'):
        self.detector_type = detector_type
        if detector_type == "mtcnn":
            # self.detector = MtcnnDetector(model_folder='./mtcnn-model',
            #      minsize=20,
            #      threshold=[0.6, 0.7, 0.8])
            pass
        else:
            print("Use retinaface")
            if use_large_detector:
                self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
            else:
                self.detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
            self.detector.prepare(ctx_id=ctx_id)

        if ctx_id>=0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        image_size = (112, 112)
        self.model = get_model(ctx, image_size, model_prefix, model_epoch, 'fc1')
        self.image_size = image_size

    def detect_and_align(self, face_img, img_detect_size):

        face_img = square_crop(face_img, img_detect_size)

        if self.detector_type == "mtcnn":
            ret = self.detector.detect_face(face_img)
            print("RET: ", ret)
            if ret is None:
                print("No image found")
                return None
            bbox, points = ret
            print("bbox: ", bbox, np.shape(bbox[:, 0])[0])
            print("points: ", points, np.shape(points[:, 0]))
            if bbox.shape[0] == 0:
                return None


            bbox = bbox[0, 0:4]
            pts5 = points[0, :].reshape((2, 5)).T
        else:
            bbox, pts5 = self.detector.detect(face_img, threshold=0.8)
            if bbox.shape[0] == 0:
                return None
            # print(bbox)
            idx_max_box = self.get_largest_face(bbox)
            bbox = bbox[idx_max_box, 0:4]


            # head_img = self.extend_and_crop_head(face_img, bbox, ratio_extend=0.4)
            # bbox, pts5 = self.detector.detect(head_img, threshold=0.8)
            # if bbox.shape[0] == 0:
            #     return None
            pts5 = pts5[idx_max_box, :]

        head_img = face_img
        nimg = face_align.norm_crop(head_img, pts5)

        img_raw = head_img.copy()
        for b in pts5:
                # landms
                cv2.circle(img_raw, (b[0], b[1]), 1, (0, 0, 255), 4)

            # save image

        name = "result.jpg"
        cv2.imwrite(name, img_raw)
        return nimg

    def extend_and_crop_head(self, img, boxes, ratio_extend=0.05):
        x_min, y_min, x_max, y_max = boxes.astype("int")
        w, h = x_max - x_min, y_max - y_min
        img_size = img.shape  # h, w, channel
        x_min, y_min, x_max, y_max = int(max(0, x_min - w * ratio_extend)), int(max(0, y_min - h * ratio_extend)), int(min(img_size[1], x_max + w*ratio_extend)), int(min(img_size[0], y_max + h* ratio_extend))
        print(x_min, y_min, x_max, y_max)
        head_img = img[y_min:y_max, x_min:x_max]
        return  head_img

    def get_largest_face(self, boxes):
        max_idx = 0
        max_s = -10
        for i in range(boxes.shape[0]):
            box = boxes[i, 0:4]
            x_min, y_min, x_max, y_max = box
            s = (x_max - x_min) * (y_max - y_min)
            if s > max_s:
                max_idx = i
                max_s = s

        return max_idx



    def get_feature(self, aligned):
        a = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()[0]
        # L2 normalize feature
        norm = np.sqrt(np.sum(emb*emb)+0.00001)
        emb /= norm

        # tuong tu
        # emb = sklearn.preprocessing.normalize(emb)
        return emb


