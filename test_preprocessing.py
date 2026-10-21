"""Compare actual model inputs, and verify inverse box geometry on varied shapes."""
from pathlib import Path
import tempfile
import unittest

from runtime import load_config
from smoke_test import make_fixture
from dataset import stage_dataset
from modules import build_detector
from evaluation import inference_args, validator_class


class PreprocessingTests(unittest.TestCase):
    def test_validation_equals_prediction_and_boxes_return_to_original_pixels(self):
        import torch
        from PIL import Image, ImageDraw
        from ultralytics.cfg import get_cfg
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.models.yolo.detect import DetectionPredictor
        from ultralytics.utils import ops
        from ultralytics.utils.patches import imread
        from preprocessing import build_project_dataset
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, prepared = make_fixture(Path(__file__).with_name('experiment.json'), root / 'fixture')
            shapes = [(200, 133), (131, 257), (31, 19), (128, 128)]
            for i, shape in enumerate(shapes):
                im = Image.new('RGB', shape, (30+i*20, 140, 210))
                draw = ImageDraw.Draw(im); draw.line((0, 0, shape[0]-1, shape[1]-1), fill=(240, 10, 90), width=3)
                im.save(root / f'fixture/images/SYNTH_1_{i:03d}.png')
            stage = root / 'stage'; stage_dataset(prepared, config, stage)
            cfg = load_config(config); data = check_det_dataset(str(stage / 'trainval.yaml'))
            predictor = DetectionPredictor(overrides={**inference_args(cfg, 'cpu'), 'save': False})
            predictor.setup_model(model=build_detector('yolov8n.yaml'), verbose=False)
            for size in (128, 640):
                args = get_cfg(overrides={**cfg['train'], 'task': 'detect', 'imgsz': size})
                ds = build_project_dataset(args, data['val'], 2, data, 'val')
                predictor.imgsz = [size, size]
                for i, path in enumerate(ds.im_files):
                    sample = ds[i]
                    evaluated = sample['img'].float()/255
                    predicted = predictor.preprocess([imread(path)])[0]
                    self.assertTrue(torch.equal(evaluated, predicted), (size, path))
                    xyxy = ops.xywh2xyxy(sample['bboxes'].clone()) * size
                    original = ops.scale_boxes((size, size), xyxy, sample['ori_shape'], sample['ratio_pad'])
                    h, w = sample['ori_shape']
                    wanted = torch.tensor([[.3*w, .25*h, .7*w, .75*h]])
                    self.assertTrue(torch.allclose(original, wanted, atol=1e-4, rtol=0), (original, wanted))

    def test_single_label_class_aware_nms(self):
        import torch
        from ultralytics.cfg import get_cfg
        validator = validator_class()(args=get_cfg(overrides={'conf': .01, 'iou': .7, 'max_det': 300}))
        validator.nc = 2
        raw = torch.tensor([[[50., 50., 40., 40., .9, .8],
                             [50., 50., 40., 40., .7, .6],
                             [50., 50., 40., 40., .1, .85]]]).transpose(1, 2)
        output = validator.postprocess(raw)[0]
        self.assertEqual(output['cls'].tolist(), [0., 1.])
        self.assertTrue(torch.allclose(output['conf'], torch.tensor([.9, .85])))


if __name__ == '__main__':
    unittest.main()
