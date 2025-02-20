from typing import *
import os
import ImageProcessorsModule.MakeTFRecordProcessors as Processors
import ImageProcessorsModule.TFRecordWriter as RecordWriter


def return_processors():
    train_processors = []
    """
    First, load in all of the files and name them as ct_handle and dose_handle
    """
    train_processors += [
        Processors.LoadNifti(nifti_path_keys=('ct_file_path', 'mask_file_path'), out_keys=('ct_handle', 'mask_handle'))
    ]
    """
    Place holder if you want to resample
    """
    train_processors += [
        Processors.ResampleSITKHandles(desired_output_spacing=(1.25, 1.25, 3.0),
                                       resample_keys=('ct_handle', 'mask_handle'),
                                       resample_interpolators=('Linear', 'Nearest'))
    ]
    """
    Now, we need to convert them to NumPy arrays for deep learning
    """
    train_processors += [
        Processors.SimpleITKImageToArray(nifti_keys=('ct_handle', 'mask_handle'), out_keys=('ct_array', 'mask_array'),
                                         dtypes=['float32', 'int'])
    ]
    """
    Remove the handles for record writing, I do not want to hold on to the SimpleITK Images
    """
    train_processors += [
        Processors.DeleteKeys(keys_to_delete=('ct_handle', 'mask_handle'))
    ]
    """
    I want to make a 3D UNet, so I'll need my images to have dimensions that are divisible by 2^(layers deep)
    You can 'box' around an annotation, but because my power_val_c and power_val_r are 512, it will just make it 512
    """
    train_processors += [
        Processors.Box_Images(image_keys=('ct_array', 'mask_array'), annotation_key='mask_array',
                              wanted_vals_for_bbox=[1], bounding_box_expansion=(10, 20, 20),
                              power_val_z=64, power_val_c=256, power_val_r=256)
    ]
    """
    Distribute into 3D images. This is for a 3D UNet, you can also distribute into 2D which will make lots of 2D images
    """
    train_processors += [
        Processors.Distribute_into_3D(image_keys=('ct_array', 'mask_array'))
    ]
    return train_processors


def make_list_dictionary(base_path):
    output_list = []
    for root, directories, files in os.walk(base_path):
        """
        Iterate through the folders and check and see where we have an Image and Mask file
        """
        if 'Image.nii.gz' not in files or 'Mask.nii.gz' not in files:
            continue
        patient_dict = {'ct_file_path': os.path.join(root, 'Image.nii.gz'),
                        'mask_file_path': os.path.join(root, 'Mask.nii.gz'),
                        'out_file_name': f"{os.path.split(root)[1]}_Record.tfrecord"}
        output_list.append(patient_dict)
    return output_list


def make_records(base_path: Union[str, bytes, os.PathLike],
                 out_path: Union[str, bytes, os.PathLike], rewrite=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    train_list = make_list_dictionary(base_path)
    """
    This makes a List of Dictionaries!
    Each entry has a ct_file_path, mask_file_path, and an out_file_name
    """
    print(train_list)
    print(train_list[0])
    """
    Lets make some processors
    """
    train_processors = return_processors()
    record_writer = RecordWriter.RecordWriter(out_path=out_path,
                                              file_name_key='out_file_name', rewrite=rewrite)
    RecordWriter.parallel_record_writer(dictionary_list=train_list, recordwriter=record_writer, thread_count=1,
                                        image_processors=train_processors, debug=True, verbose=False)  # thread_count=1,
    return None


if __name__ == '__main__':
    pass
