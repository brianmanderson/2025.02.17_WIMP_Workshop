from tqdm import tqdm
from DicomRTTool.ReaderWriter import DicomReaderWriter, sitk
import os


def dicom_to_nifti(dicom_top_path):
    directories_to_run = []
    """
    Walk through the folders and see if we have DICOM present
    """
    for root, directories, files in os.walk(dicom_top_path):
        if len([i for i in files if i.endswith('.dcm')]) == 0:
            continue
        if 'Image.nii.gz' in files:
            continue
        directories_to_run.append(root)
    pbar = tqdm(total=len(directories_to_run), desc='Writing images')
    for root in directories_to_run:
        reader = DicomReaderWriter(verbose=False, Contour_Names=['ctv_pelvis'])
        reader.walk_through_folders(root)
        reader.get_images_and_mask()
        sitk.WriteImage(reader.dicom_handle, os.path.join(root, f"Image.nii.gz"))
        sitk.WriteImage(reader.annotation_handle, os.path.join(root, f"Mask.nii.gz"))
        pbar.update()


if __name__ == '__main__':
    pass
