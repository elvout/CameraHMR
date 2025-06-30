import argparse
from mesh_estimator import HumanMeshEstimator2


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    parser.add_argument("--image_folder", "--image_folder", type=str,
        help="Path to input image folder.")
    parser.add_argument("--output_folder", "--output_folder", type=str,
        help="Path to folder output folder.")
    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator2()
    estimator.run_on_video(
        "Meta_G201_202_Columns_Quickly_round10_rep2_n_view0_us1416148067.mp4"
    )

if __name__=='__main__':
    main()
