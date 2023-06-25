import argparse
from collections import deque
import json
import cv2
import numpy as np
from tqdm import trange
from ultralytics import YOLO
import albumentations as alb


FRAME_WIDTH, FRAME_HEIGHT = 640, 360
BUFFER_SIZE = 20
SPECIAL_EQUIPMENT_CLASSES = {
                                0: 'подъёмный кран',
                                1: 'грузовой автомобиль',
                                2: 'экскаватор',
                                3: 'трактор',
                                4: 'каток',
                                5: 'грузовичок (газель)'
                             }

# SPECIAL_EQUIPMENT_CLASSES = {
#                                 0: 'crane',
#                                 1: 'truck',
#                                 2: 'excavator',
#                                 3: 'tractor',
#                                 4: 'roller',
#                                 5: 'cargo'
#                             }

EVENTS = {0: 'работает', 1: 'простой'}


def parse_args():
    parser = argparse.ArgumentParser(description="2023year predict events from video")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("-w", "--weights", default=None, help="Detection model weights file")
    parser.add_argument("-wcl", "--weights_cl", default=None, help="Classification model weights file")
    parser.add_argument("-fw", "--frame_width", type=int, default=FRAME_WIDTH, help=f"Frame width (default={FRAME_WIDTH})")
    parser.add_argument("-fh", "--frame_height", type=int, default=FRAME_HEIGHT, help=f"Frame height (default={FRAME_HEIGHT})")
    parser.add_argument("-s", "--sec", default=None, help="Sec to read (int) (default=None)")
    parser.add_argument("-o", "--out_video_filename", default=None, help="Output video filename (default=None)")
    parser.add_argument("-j", "--out_json_filename", default='result.json', help="Output json filename (default='result.json')")

    args = parser.parse_args()
    return args


def predict_shift(model_cls, buffer_memory):
    coord = np.array([x1y1x2y2 for _, x1y1x2y2, _ in buffer_memory])
    patches = [patche for _, _, patche in buffer_memory]
    x1, y1, x2, y2 = coord[:, 0].min(), coord[:, 1].min(), coord[:, 2].max(), coord[:, 3].max()
    dx, dy = x1, y1
    w, h = x2-x1+10, y2-y1+10
    shift_img = np.ones((BUFFER_SIZE, h,w), dtype=np.uint8) * 255
    for idx, patch in enumerate(patches[:40]):
        px1, py1, px2, py2 = coord[idx, 0] - dx, coord[idx, 1] - dy, coord[idx, 2] - dx, coord[idx, 3] - dy
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            shift_img[idx, py1:py2, px1:px2] = patch.squeeze()
    shift_img = shift_img.mean(axis=0).astype(np.uint8)
    shift_img = cv2.cvtColor(shift_img, cv2.COLOR_GRAY2RGB)

    results = model_cls(shift_img, verbose=False)
    return results


def main():
    args = parse_args()

    if args.weights is None:
        args.weights = 'weights/yolov8s_50ep_bs16.pt'

    if args.weights_cl is None:
        args.weights_cl = 'weights/yolov8n_cls.pt'

    if args.sec is not None:
        args.sec = int(args.sec)

    model = YOLO(args.weights)
    model_cls = YOLO(args.weights_cl)

    transform = alb.Compose([alb.Resize(args.frame_height, args.frame_width, cv2.INTER_LANCZOS4)])

    cap = cv2.VideoCapture(args.video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Frame_count: {frame_count}')
    print(f'FPS : {fps}')

    outer = None
    if args.out_video_filename is not None:
        fps25 = 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outer = cv2.VideoWriter(args.out_video_filename, fourcc, fps25, (args.frame_width, args.frame_height))

    if args.sec is not None:
        frame_count = min(frame_count, args.sec * fps)

    json_d = {}
    equipments = {}
    for i in trange(0, frame_count, fps):
        # cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame = np.array(transform(image=frame)["image"])
            results = model.track(source=frame, persist=True, verbose=False)

            boxes = results[0].boxes.xyxy.numpy().astype(int)
            cls = results[0].boxes.cls.numpy().astype(int)
            ids = results[0].boxes.id.numpy().astype(int)
        
            img_gray = np.expand_dims(results[0].orig_img.mean(axis=2), axis=2)

            for box, cl, id_track in zip(boxes, cls, ids):
                if id_track not in equipments:
                    equipments[id_track] = deque(maxlen=BUFFER_SIZE)
                patch = img_gray[box[1]:box[3], box[0]:box[2]].copy()
                equipments[id_track].append(((id_track, cl), box, patch))
                pred = None
                object_state = None
                if len(equipments[id_track]) == BUFFER_SIZE:
                    pred = predict_shift(model_cls, equipments[id_track])
                    object_state = np.argmax(pred[0].probs.data.cpu().numpy())

                if str(id_track) not in json_d:
                    json_d[str(id_track)] = {
                                        'class': SPECIAL_EQUIPMENT_CLASSES[cl],
                                        'inplace': [
                                            {
                                                'begin': i // fps,
                                                'end': i // fps
                                            }
                                        ],
                                        'downtime': []
                                        }
                else:
                    json_d[str(id_track)]['inplace'][-1]['end'] = i // fps

                if pred is not None and object_state is not None:
                    if object_state == 1:
                        if len(json_d[str(id_track)]['downtime']) == 0:
                            json_d[str(id_track)]['downtime'].append({
                                'begin': i // fps - (BUFFER_SIZE - 1),
                                'end': i // fps
                            })
                        else:
                            last = json_d[str(id_track)]['downtime'][-1]
                            if last['end'] >= i // fps - (BUFFER_SIZE - 1):
                                last['end'] = i // fps
                            else:
                                json_d[str(id_track)]['downtime'].append({
                                    'begin': i // fps - (BUFFER_SIZE - 1),
                                    'end': i // fps
                                })

            if outer is not None:
                frame = results[0].plot(conf=True, font_size=None, line_width=1)
                outer.write(frame)

        else:
            print('>>> ERROR: Can not read data!')
            break

    cap.release()
    if outer is not None:
        outer.release()

    with open(args.out_json_filename, 'wt', encoding='utf8') as f:
        json.dump({'events': json_d}, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
