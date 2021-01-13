import cv2

from face_model import FaceModel
import time
import numpy as np


class MatchingFaceModel():
    def __init__(self, image_size = (112, 112), model_path= './weights/model-r100-ii/model,0', scale=640, gpu=-1, use_large_detector=False):
        self.image_size = image_size
        self.model_path = model_path
        self.gpu = gpu

        vec = model_path.split(',')
        model_prefix = vec[0]
        model_epoch = int(vec[1])
        self.face_model = FaceModel(gpu, model_prefix, model_epoch, use_large_detector=use_large_detector, detector_type='retina')
        self.scale = scale

    def matching(self, img1, img2):
        t1 = time.time()
        img1_aligned = self.face_model.detect_and_align(img1, img_detect_size=self.scale)
        print("Time detect and align: %.2f" % (time.time() - t1))
        img2_aligned = self.face_model.detect_and_align(img2, img_detect_size=self.scale)
        if img1_aligned is not None and img2_aligned is not None:
            t2 = time.time()
            f1 = self.face_model.get_feature(img1_aligned)
            print("Time get feature: %.2f" % (time.time() - t2))

            f2 = self.face_model.get_feature(img2_aligned)

            sim = self.findCosineDistance(f1, f2)
            print("Cosine distance: ", sim)
            return sim, img1_aligned, img2_aligned
        else:
            return None, None, None

    def square_crop(self, im, S):
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
        return det_im, scale

    def findCosineDistance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def convert_to_percentage(self, cosine_score, min_val=0.0, max_val=2.0, cosine_threshold=0.75):
        percentage = 100 - (cosine_score - min_val) / (max_val - min_val) * 100
        percentage_threshold = 100 - (cosine_threshold - min_val) / (max_val - min_val) * 100
        return percentage, percentage_threshold


if __name__ == "__main__":
    matching_model = MatchingFaceModel(gpu=0)
    img1 = cv2.imread('imgs/hardest_cmnd.jpg')
    img2 = cv2.imread('imgs/hardest_selfie.jpg')
    matching_model.matching(img1, img2)