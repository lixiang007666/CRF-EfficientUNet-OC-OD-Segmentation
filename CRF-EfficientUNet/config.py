import glob
import os
import random
from collections import defaultdict

from albumentations import (
    CLAHE, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, OneOf, Compose
)

__all__ = ['DEFAULT_CONFIG']

VERBOSE = True

# Echocardio parameter
CHAMBER = "2C"
DATE = "2019-08-21"

# Parameters
TRAIN_TEST_SPLIT_FRACTION = 0.8
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400

# EDIT: set data path here
TEST_DATA = f'datasets/{DATE}_{CHAMBER}/test/'
TRAIN_DATA = f'datasets/{DATE}_{CHAMBER}/train_dev/'
SOURCE_DICOM_DATA_PATH = f"data_dicom/{DATE}/"
DEBUG_DATA = "data/saved_sample"


# data augmentation
def strong_aug(p=0.5):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


AUGMENTATION = strong_aug(p=0.0)

# construct right path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(os.path.abspath(os.path.join(ROOT_DIR, os.pardir)), TEST_DATA)
TRAIN_DATA_PATH = os.path.join(os.path.abspath(os.path.join(ROOT_DIR, os.pardir)), TRAIN_DATA)
SOURCE_DICOM_DATA_PATH = os.path.join(os.path.abspath(os.path.join(ROOT_DIR, os.pardir)), SOURCE_DICOM_DATA_PATH)
DEBUG_DATA_PATH = os.path.join(os.path.abspath(os.path.join(ROOT_DIR, os.pardir)), DEBUG_DATA)


class DEFAULT_CONFIG:
    VERBOSE = VERBOSE
    TEST_DATA_PATH = TEST_DATA_PATH
    TRAIN_DATA_PATH = TRAIN_DATA_PATH
    IMG_WIDTH = IMG_WIDTH
    IMG_HEIGHT = IMG_HEIGHT
    BUFFER_SIZE = BUFFER_SIZE
    BATCH_SIZE = BATCH_SIZE
    SOURCE_DICOM_DATA_PATH = SOURCE_DICOM_DATA_PATH
    TRAIN_TEST_SPLIT_FRACTION = TRAIN_TEST_SPLIT_FRACTION
    DEBUG_DATA_PATH = DEBUG_DATA_PATH
    CHAMBER = CHAMBER
    AUGMENTATION = AUGMENTATION


def get_patient(file, chamber_to_get):
    if os.path.exists(file + ".json"):
        dirname = os.path.dirname(file)
        chamber = os.path.basename(dirname)
        if chamber == chamber_to_get:
            dirpatient = os.path.dirname(dirname)
            return dirpatient
    return None


if __name__ == '__main__':
    root = "/home/gungui/PycharmProjects/aicardio/aicardio/data_dicom/2019-08-21/"
    dicom_paths = defaultdict(list)
    for file in glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True):
        # print(file)
        patient = get_patient(file, '2C')
        # print("patient=", patient)
        if patient is not None:
            dicom_paths[patient].append(file)
    patients = list(dicom_paths.keys())
    random.shuffle(patients)
    n = len(patients)
    train_patients, test_patients = patients[:n * 4 // 5], patients[n * 4 // 5:]
    train_dicoms = sum([dicom_paths[p] for p in train_patients], [])
    test_dicoms = sum([dicom_paths[p] for p in test_patients], [])

    print(test_dicoms)

    # for fps_folder in os.listdir('/home/gungui/PycharmProjects/aicardio/aicardio/data_dicom/2019-08-21/'):
    #     fps_folder = os.path.join(root, fps_folder)
    #     for patent in os.listdir(os.path.join(root, fps_folder)):
    #         print(os.path.join(fps_folder, patent))
