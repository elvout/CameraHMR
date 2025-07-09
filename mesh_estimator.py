import os
from glob import glob
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import smplx
import torch
import tqdm
import trimesh
import yacs.config
from detectron2.config import LazyConfig
from elvout.pyu.video_iterator import VideoIterator
from torchvision.transforms import Normalize

from core.cam_model.fl_net import FLNet
from core.camerahmr_model import CameraHMR
from core.constants import (
    CAM_MODEL_CKPT,
    CHECKPOINT_PATH,
    DETECTRON_CFG,
    DETECTRON_CKPT,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    NUM_BETAS,
    SMPL_MODEL_PATH,
)
from core.datasets.dataset import Dataset
from core.utils import recursive_to
from core.utils.renderer_pyrd import Renderer
from core.utils.utils_detectron2 import DefaultPredictor_Lazy


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y : start_y + new_height, start_x : start_x + new_width] = (
        resized_img
    )

    return aspect_ratio, final_img


class Context:
    def __init__(self, trial_base_name: str) -> None:
        """
        Semantic parsing of the Freeman trajectory naming convention.

        General form:
        Meta_G###_###_{obstacle}_{modus}_round##_rep#_{codes}
        """

        self.trial_base_name: str
        self.session: str
        # TODO(elvout): switch to enums?
        self.obstacle: str
        self.modus: str
        self.round: int
        self.codes: str
        ###############################################################

        self.trial_base_name = trial_base_name
        tokens = trial_base_name.split("_")
        self.session = tokens[1] + "_" + tokens[2]
        self.obstacle = tokens[3]
        self.modus = tokens[4]
        self.round = int(tokens[5][5:])
        self.codes = tokens[7]


class VideoContext(Context):
    def __init__(self, name: str) -> None:
        """
        Semantic parsing of the Freeman trajectory naming convention, extended
        for video views.

        General form:
        Meta_G###_###_{obstacle}_{modus}_round##_rep#_{codes}_view#_us##########
        """
        super().__init__(name[: name.find("_view")])

        self.view: int
        self.start_time: float
        ###############################################################

        tokens = name.split("_")
        assert tokens[8].startswith("view")
        self.view = int(tokens[8][4:])
        assert tokens[9].startswith("us")
        self.start_time = float(tokens[9][2:]) * 1e-6


class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(
            model_path=smpl_model_path, num_betas=NUM_BETAS
        ).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        torch.serialization.add_safe_globals([yacs.config.CfgNode])
        checkpoint = torch.load(CAM_MODEL_CKPT)["state_dict"]
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def init_detector(self, threshold):
        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[
                i
            ].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    def convert_to_full_img_cam(
        self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length
    ):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2.0 * focal_length / (bbox_height * s)
        cx = 2.0 * (bbox_center[:, 0] - (img_w / 2.0)) / (s * bbox_height)
        cy = 2.0 * (bbox_center[:, 1] - (img_h / 2.0)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch["img_size"][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch["box_size"],
            bbox_center=batch["box_center"],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch["cam_int"][:, 0, 0],
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = (
            np.transpose(img_full_resized.astype("float32"), (2, 0, 1)) / 255.0
        )
        img_full_resized = self.normalize_img(
            torch.from_numpy(img_full_resized).float()
        )

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array(
            [[fl_h, 0, img_w / 2], [0, fl_h, img_h / 2], [0, 0, 1]]
        ).astype(np.float32)
        return cam_int

    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)

    def process_image(self, img_path, output_img_folder, i):
        img_cv2 = cv2.imread(str(img_path))
        # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(
            output_img_folder, f"{os.path.basename(fname)}_{i:06d}{img_ext}"
        )
        smpl_fname = os.path.join(
            output_img_folder, f"{os.path.basename(fname)}_{i:06d}.smpl"
        )
        mesh_fname = os.path.join(
            output_img_folder, f"{os.path.basename(fname)}_{i:06d}.obj"
        )

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Get Camera intrinsics using HumanFoV Model
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=10
        )

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch["img_size"][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(
                out_smpl_params, out_cam, batch
            )

            mesh = trimesh.Trimesh(
                output_vertices[0].cpu().numpy(), self.smpl_model.faces, process=False
            )
            mesh.export(mesh_fname)

            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = (
                (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            )
            renderer = Renderer(
                focal_length=focal_length[0],
                img_w=img_w,
                img_h=img_h,
                faces=self.smpl_model.faces,
                same_mesh_color=True,
            )
            front_view = renderer.render_front_view(
                pred_vertices_array, bg_img_rgb=img_cv2.copy()
            )
            final_img = front_view
            # Write overlay
            cv2.imwrite(overlay_fname, final_img)
            renderer.delete()

    def run_on_images(self, image_folder, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        ]
        images_list = [
            image
            for ext in image_extensions
            for image in glob(os.path.join(image_folder, ext))
        ]
        for ind, img_path in enumerate(tqdm.tqdm(images_list, ncols=80)):
            self.process_image(img_path, out_folder, ind)


class HumanMeshEstimator2(HumanMeshEstimator):
    def __init__(self) -> None:
        super().__init__()

    def convert_to_full_img_cam(
        self, pare_cam, bbox_height, bbox_center, int_cx, int_cy, focal_length
    ):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2.0 * focal_length / (bbox_height * s)
        cx = 2.0 * (bbox_center[:, 0] - int_cx) / (s * bbox_height)
        cy = 2.0 * (bbox_center[:, 1] - int_cy) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch["box_size"],
            bbox_center=batch["box_center"],
            int_cx=batch["cam_int"][:, 0, 2],
            int_cy=batch["cam_int"][:, 1, 2],
            focal_length=batch["cam_int"][:, 0, 0],
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    # TODO(elvout): batching
    # https://github.com/facebookresearch/detectron2/issues/282
    def frame_dataset(
        self,
        frame: npt.NDArray[np.uint8],
        frame_number: int,
        intrinsics_matrix: npt.NDArray[np.float32] | None = None,
    ) -> Dataset:
        # Detect humans in the image
        det_out = self.detector(frame)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        if intrinsics_matrix is None:
            intrinsics_matrix = self.get_cam_intrinsics(frame)

        return Dataset(
            frame, bbox_center, bbox_scale, intrinsics_matrix, False, f"{frame_number}"
        )

    def process_dataset(
        self,
        dataset: Dataset | torch.utils.data.ConcatDataset,
        output_folder: Path,
    ) -> None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=1,
        )

        frame_numbers = []
        joints = []
        cam_trans = []

        for batch in tqdm.tqdm(dataloader, ncols=80):
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch["img_size"][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(
                out_smpl_params, out_cam, batch
            )

            frame_numbers.append([int(s) for s in batch["imgname"]])
            joints.append(output_joints.detach().cpu().numpy())
            cam_trans.append(output_cam_trans.detach().cpu().numpy())

        output_folder.mkdir(parents=True, exist_ok=True)

        _fn = np.concat(frame_numbers)
        np.save(output_folder / "frame_numbers.npy", _fn, allow_pickle=False)
        _j = np.concat(joints, axis=0)
        np.save(output_folder / "joints.npy", _j, allow_pickle=False)
        _ct = np.concat(cam_trans, axis=0)
        np.save(output_folder / "cam_trans.npy", _ct, allow_pickle=False)

    def process_frame(
        self,
        frame: npt.NDArray[np.uint8],
        frame_number: int,
        intrinsics_matrix: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.uint8]:
        # Detect humans in the image
        det_out = self.detector(frame)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # with open("results/scores.txt", "a") as fout:
        #     fout.write(f"{frame_number}: {det_instances.scores[valid_idx]}\n{bbox_center}\n")

        if intrinsics_matrix is None:
            intrinsics_matrix = self.get_cam_intrinsics(frame)
        # TODO(elvout): Create the dataset using all frames (e.g., ConcatDataset)
        # to parallelize inference across frames? Fill in the img_path parameter
        # with the frame number.
        dataset = Dataset(
            frame, bbox_center, bbox_scale, intrinsics_matrix, False, f"{frame_number}"
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=1,
        )

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch["img_size"][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(
                out_smpl_params, out_cam, batch
            )

            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = (
                (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            )
            renderer = Renderer(
                focal_length=focal_length[0],
                img_w=img_w,
                img_h=img_h,
                cx=batch["cam_int"][0, 0, 2],
                cy=batch["cam_int"][0, 1, 2],
                faces=self.smpl_model.faces,
                same_mesh_color=True,
            )
            front_view = renderer.render_front_view(
                pred_vertices_array, bg_img_rgb=frame.copy()
            )
            final_img = front_view
            # Write overlay
            renderer.delete()
        return final_img

    # NOTE: parent class code indicates that the model may have been trained on BGR.
    def run_on_video(self, video_path: str | Path) -> None:
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        output_folder = Path(f"results/{video_path.stem}")
        if output_folder.exists():
            # Assume results are valid
            return

        datasets = []
        context = VideoContext(video_path.stem)
        intrinsics_matrix = np.load(
            f"freeman-intrinsics/{context.session}_view{context.view}_undistorted_calib.npy"
        )

        video_iterator = VideoIterator(video_path)
        for frame_number, frame in tqdm.tqdm(video_iterator, ncols=80):
            datasets.append(
                self.frame_dataset(frame[:, :, ::-1], frame_number, intrinsics_matrix)
            )

        self.process_dataset(
            torch.utils.data.ConcatDataset(datasets),
            output_folder,
        )

        # Debug visualization
        if False:
            video_iterator.reset(0)
            output_folder.mkdir(parents=True, exist_ok=True)
            with iio.imopen(
                str(output_folder / "vis.mp4"), "w", plugin="pyav"
            ) as out_vid:
                out_vid.init_video_stream("h264", fps=30)

                for frame_number, frame in tqdm.tqdm(video_iterator, ncols=80):
                    # if frame_number > 60:
                    #     continue
                    out_vid.write_frame(
                        self.process_frame(
                            frame[:, :, ::-1], frame_number, intrinsics_matrix
                        )
                    )
