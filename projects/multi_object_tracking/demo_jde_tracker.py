import argparse
import torch
import cv2

from detector_utils import *

from mot.tracker import JDETracker

class DemoJDETracker():
    def __init__(self, opt):
        self.opt = opt
        cfg_dict = parse_model_cfg(opt.cfg)
        self.opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

        self.result_root = self.opt.output_root if opt.output_root != '' else '.'
        os.makedirs(self.result_root, exist_ok=True)
        self.frame_dir = os.path.join(self.result_root, 'frame')
        os.makedirs(self.frame_dir, exist_ok=True)

        self.dataloader = LoadVideo(opt.input_video, opt.img_size)

        self.model = Darknet(opt.cfg, nID=14455)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)

        if torch.cuda.is_available():
            self.model.cuda().eval()
        else:
            self.model.eval()

        print("Model load complete")

        self.tracker = JDETracker()

    def track_video(self):
        results = []
        frame_id = 0

        for path, img, img0 in self.dataloader:
            im_blob = torch.from_numpy(img).cuda().unsqueeze(0)

            with torch.no_grad():
                pred_model = self.model(im_blob)

            if len(pred_model > 0):
                #  perform additional steps before pushing into the tracker
                #  such as threshold filtering, NMS and scaling coordinates
                pred_model = pred_model[pred_model[:, :, 4] > self.opt.conf_thres]
                pred_model = non_max_suppression(pred_model.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
                scale_coords(self.opt.img_size, pred_model[:, :4], img0.shape).round()

                # split the pred_model into two parts -
                # pred_dets giving detection results of image -  i.e. (batch_id, x1, y1, x2, y2, object_conf)
                # pred_embs giving Embedding results of the image
                pred_dets, pred_embs = pred_model[:, :5], pred_model[:, 6:]
                online_targets = self.tracker.update(pred_dets, pred_embs)

                # saving the results for display
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                online_im = plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id)
                cv2.imwrite(os.path.join(self.frame_dir, '{:05d}.jpg'.format(frame_id)), online_im)

            frame_id += 1
            print(frame_id)
            if(frame_id == 20):
                break

        # save results as video after the loop
        if self.opt.output_format == 'video':
            output_video_path = os.path.join(self.result_root, 'result.mp4')
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(self.frame_dir, output_video_path)
            os.system(cmd_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--input-video', type=str, help='path to the input video')
    parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    jde_obj = DemoJDETracker(opt)
    jde_obj.track_video()