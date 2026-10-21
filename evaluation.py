"""Operational detection metrics, separate from image-level malignant alerts.

One GT per image; confidence-ordered, class-correct, one-to-one matching.
AP remains the pinned library's implementation, using the same NMS class policy.
"""
import math
from runtime import NAMES


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def gt_record(row):
    x, y, w, h = [float(row[k]) for k in ('center_x', 'center_y', 'box_width', 'box_height')]
    return {'image_id': row['image_id'], 'class_id': int(row['class_id']),
            'box': [x-w/2, y-h/2, x+w/2, y+h/2]}


def safe_ratio(a, b):
    return a / b if b else None


def score_records(records, threshold, match_iou=0.5):
    if not 0 <= threshold <= 1 or not 0 < match_iou <= 1:
        raise ValueError('Invalid score/IoU threshold')
    counts = {c: {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0} for c in NAMES}
    overlaps = {c: {'all_gt_class_correct': [], 'all_gt_class_agnostic': [], 'tp_only': []} for c in NAMES}
    alerts = {'malignant_tp': 0, 'malignant_fn': 0, 'benign_fp': 0, 'benign_tn': 0}
    seen = set()
    for record in records:
        identifier = record['image_id']; c = record['class_id']; box = record['box']
        if identifier in seen or c not in NAMES:
            raise ValueError('Duplicate GT image or invalid GT class')
        seen.add(identifier); counts[c]['gt'] += 1
        predictions = []
        for p in record['predictions']:
            values = [p['confidence'], *p['box']]
            if (p['class_id'] not in NAMES or not all(map(math.isfinite, values))
                    or not 0 <= p['confidence'] <= 1 or len(p['box']) != 4
                    or p['box'][2] < p['box'][0] or p['box'][3] < p['box'][1]):
                raise ValueError('Invalid prediction')
            if p['confidence'] > threshold:
                predictions.append(p)
        predictions.sort(key=lambda p: -p['confidence'])
        matched = False
        for p in predictions:
            pc = p['class_id']; overlap = iou(box, p['box'])
            if not matched and pc == c and overlap >= match_iou:
                counts[c]['tp'] += 1; matched = True
                overlaps[c]['tp_only'].append(overlap)
            else:
                counts[pc]['fp'] += 1
        counts[c]['fn'] += int(not matched)
        overlaps[c]['all_gt_class_correct'].append(max((iou(box, p['box']) for p in predictions if p['class_id'] == c), default=0.0))
        overlaps[c]['all_gt_class_agnostic'].append(max((iou(box, p['box']) for p in predictions), default=0.0))
        alarm = any(p['class_id'] == 1 for p in predictions)
        alerts[('malignant_tp' if alarm else 'malignant_fn') if c == 1 else ('benign_fp' if alarm else 'benign_tn')] += 1
    classes = {}
    for c, values in counts.items():
        classes[NAMES[c]] = {**values, 'precision': safe_ratio(values['tp'], values['tp']+values['fp']),
                             'recall': safe_ratio(values['tp'], values['gt']),
                             'iou': {key: {'mean': safe_ratio(sum(v), len(v)), 'denominator': len(v)}
                                     for key, v in overlaps[c].items()}}
    alerts['sensitivity'] = safe_ratio(alerts['malignant_tp'], alerts['malignant_tp']+alerts['malignant_fn'])
    alerts['benign_false_alert_rate'] = safe_ratio(alerts['benign_fp'], alerts['benign_fp']+alerts['benign_tn'])
    return {'threshold': threshold, 'score_rule': 'confidence > threshold',
            'match_iou': match_iou, 'classes': classes, 'image_alerts': alerts,
            'images': len(seen), 'image_alerts_are_biopsy_counts': False}


def select_threshold(records, policy, split='val'):
    if split != 'val':
        raise ValueError('Do not sweep thresholds on reassessment')
    scores = [score_records(records, t, policy['match_iou']) for t in policy['threshold_grid']]
    feasible = [s for s in scores if s['image_alerts']['benign_false_alert_rate'] is not None
                and s['classes']['Malignant']['recall'] is not None
                and s['image_alerts']['benign_false_alert_rate'] <= policy['benign_false_alert_budget']]
    if not feasible:
        return {'selected': None, 'grid': scores, 'reason': 'No feasible threshold in the predeclared grid'}
    def key(s):
        malignant = s['classes']['Malignant']
        return (malignant['recall'], malignant['precision'] or 0,
                -s['image_alerts']['benign_fp'], s['threshold'])
    return {'selected': max(feasible, key=key), 'grid': scores,
            'rule': 'malignant detection recall, then precision, fewer benign alerts, higher threshold'}


def validator_class():
    """Match prediction's single-label policy; default upstream val is multi-label.

    NMS adapter follows Ultralytics DetectionValidator v8.3.221 (AGPL-3.0):
    https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/models/yolo/detect/val.py
    """
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import nms

    class SingleLabelValidator(DetectionValidator):
        def build_dataset(self, img_path, mode='val', batch=None):
            from preprocessing import build_project_dataset
            return build_project_dataset(self.args, img_path, batch, self.data, mode, self.stride)

        def postprocess(self, preds):
            outputs = nms.non_max_suppression(preds, self.args.conf, self.args.iou,
                        nc=self.nc, multi_label=False, agnostic=False, max_det=self.args.max_det)
            return [{'bboxes': x[:, :4], 'conf': x[:, 4], 'cls': x[:, 5], 'extra': x[:, 6:]} for x in outputs]
    return SingleLabelValidator


def inference_args(cfg, device):
    ev = cfg['evaluation']
    return dict(imgsz=cfg['train']['imgsz'], device=device, conf=ev['score_floor'],
                iou=ev['nms_iou'], max_det=ev['max_det'], agnostic_nms=False,
                half=False, augment=False, rect=False, verbose=False)


def collect_predictions(model, rows, stage, cfg, device):
    from pathlib import Path
    records, timing = [], []
    # Use one image per call to keep prediction geometry independent of batching.
    for row in rows:
        folder = Path(stage) / 'images'
        paths = list(folder.glob(row['image_id']+'.*'))
        if len(paths) != 1:
            raise ValueError('Ambiguous staged image')
        result = model.predict(str(paths[0]), **inference_args(cfg, device))[0]
        records.append({**gt_record(row), 'predictions': [
            {'class_id': int(c), 'confidence': float(s), 'box': box}
            for c, s, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxyn.tolist())]})
        timing.append(result.speed)
    return records, timing
