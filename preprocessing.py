"""One letterbox operation for evaluation, matching the ordinary predictor.

Uses the pinned upstream LetterBox/Format implementations (AGPL-3.0).
The default YOLO validation loader first resizes with ceil, while the predictor
resizes once with round. This adapter removes that extra evaluation resize and
records the exact gain/padding for mapping boxes back to the original image.
Training still uses the declared upstream augmentation pipeline.
"""
from ultralytics.data import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils.patches import imread


def letterbox_geometry(original_shape, target_shape):
    h, w = original_shape
    th, tw = target_shape
    gain = min(th / h, tw / w)
    left = round((tw - round(w * gain)) / 2 - 0.1)
    top = round((th - round(h * gain)) / 2 - 0.1)
    return (gain, gain), (left, top)


class EvaluationLetterBox(LetterBox):
    def __call__(self, labels=None, image=None):
        original = labels['img'].shape[:2] if labels else None
        result = super().__call__(labels, image)
        if original is not None:
            result['ratio_pad'] = letterbox_geometry(original, self.new_shape)
        return result


class EvaluationDataset(YOLODataset):
    def load_image(self, i, rect_mode=True):
        # Read the original once. Do not use a pre-resized .npy cache.
        image = imread(self.im_files[i], flags=self.cv2_flag)
        if image is None:
            raise FileNotFoundError(self.im_files[i])
        if image.ndim == 2:
            image = image[..., None]
        return image, image.shape[:2], image.shape[:2]

    def build_transforms(self, hyp=None):
        if self.augment or self.rect:
            raise ValueError('EvaluationDataset requires unaugmented square batches')
        transforms = super().build_transforms(hyp)
        transforms.transforms[0] = EvaluationLetterBox(
            new_shape=(self.imgsz, self.imgsz), auto=False, scaleup=True)
        return transforms


def build_project_dataset(args, path, batch, data, mode, stride=32):
    if args.rect or args.cache:
        raise ValueError('This protocol requires square input without image caching')
    if mode == 'train':
        return build_yolo_dataset(args, path, batch, data, mode=mode, rect=False, stride=stride)
    if mode != 'val':
        raise ValueError('Unsupported loader mode')
    return EvaluationDataset(img_path=path, imgsz=args.imgsz, batch_size=batch,
            augment=False, hyp=args, rect=False, cache=None, single_cls=False,
            stride=stride, pad=0.5, prefix='val: ', task='detect', classes=None,
            data=data, fraction=1.0)
