import numpy as np
import os
import tensorflow as tf
from Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Data_Generators.Image_Processors_Module.src.Processors.TFDataSets import ConstantProcessors as CProcessors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import SimpleITK as sitk


base_data_path = os.path.join(return_records_path(), 'Data', 'TFRecords')
print(base_data_path)


def return_generator(records_path, batch=8, eval_output=False, is_val=False, image_reduction=1, auto_encoder=False,
                     **kwargs):
    generator = DataGeneratorClass(record_paths=records_path,
                                   debug=False, repeat=not is_val,
                                   shuffle=not is_val,
                                   in_parallel=2, delete_old_cache=True)
    image_keys = ('ct_array',)
    mean_value = -35
    noise = 0.5
    if is_val or eval_output:
        noise = 0.0
    standard_deviation_value = 60
    image_shape=(128, 512, 512)
    out_shape = tuple([int(i/(2**image_reduction)) for i in image_shape])
    if is_val:
        out_shape = (128, 512, 512)
    out_keys = ('mask_array',)
    min_stop_start = None
    if auto_encoder:
        out_shape = (64, 64, 64)
        min_stop_start = ((10, 128), (150, 200), (150, 250), (0, 1))
    pull_keys = image_keys
    if eval_output:
        pull_keys += ('out_file_name',)
    base_processors = [
        CProcessors.ExpandDimension(axis=-1, image_keys=image_keys + out_keys),
        CProcessors.DefineShape(keys=('ct_array', 'mask_array'), image_shapes=(image_shape + (1,),
                                                                        image_shape + (1,)))]
    if not is_val:
        base_processors += [
            CProcessors.Cast_Data(keys=('mask_array', ), dtypes=('float32',)),
            CProcessors.RandomCrop(keys_to_crop=('ct_array', 'mask_array'),
                                   crop_dimensions=out_shape + (2,),
                                   min_start_stop=min_stop_start), #
            CProcessors.FlipImages(keys=('ct_array', 'mask_array'), flip_up_down=False, flip_left_right=True,
                            on_global_3D=True, image_shape=out_shape + (2,)),
            CProcessors.Cast_Data(keys=('mask_array',), dtypes=('int32',))
                        ]
    base_processors += [
        CProcessors.CreateNewKey(input_keys=image_keys, output_keys=out_keys)
    ]
    base_processors += [
        CProcessors.Add_Constant(keys=image_keys,
                                 values=(-mean_value,)),  # Put them on a scale of roughly 0
        CProcessors.MultiplyImagesByConstant(keys=image_keys,
                                             values=(1/standard_deviation_value,)),
        CProcessors.RandomNoise(wanted_keys=('ct_array',), max_noise=noise),
        # Add -1 to make it -1 to 1
        # Now we'll threshold each of these regions to remove sections we aren't interested in
        CProcessors.Threshold_Images(keys=image_keys,
                                     lower_bounds=(-5,),
                                     upper_bounds=(5,),
                                     divides=(False,)),
    ]
    base_processors += [CProcessors.ReturnOutputs(input_keys=pull_keys, output_keys=out_keys, as_tuple=False),]
    prefetch = 15
    if is_val and not eval_output and not auto_encoder:
        prefetch = 1
        cache_path = os.path.join(return_records_path(), 'Data', 'TFRecords', 'val')
        base_processors += [{'cache': cache_path}, {'repeat'}]
    base_processors += [{'batch': batch}, {'prefetch': prefetch}, ]
    generator.compile_data_set(image_processors=base_processors, debug=False)
    return generator


def main():
    print("Getting records for train")
    train_generator = return_validation_generator(batch=1, eval_output=False, image_reduction=1)
    train_dataset = train_generator.data_set
    # print("Getting records for valid")
    # val_generator = return_validation_generator(batch=1)
    # valid_dataset = val_generator.data_set
    output_list = []
    iterator = iter(train_dataset)
    import time
    start = time.time()
    for __ in range(1):
        print(__)
        all_files = [i for i in os.listdir(base_data_path)]
        for _ in range(len(train_generator)):
            # print(_)
            x, y = next(iterator)
            print(_)
            print(x.shape)
            continue
            index = str(x[0][1]).split("'")[1].split('_')[0] + '_Record'
            # index = str(x['mask_file']).split('\\')[6]
            if _ == 0:
                print(index)
                print('--------------------')
            continue
            files = [i for i in all_files if i.startswith(index)]
            for f in files:
                os.rename(os.path.join(base_data_path, f), os.path.join(base_data_path, "All_Good", f))
            continue
            image_handle = sitk.GetImageFromArray(np.squeeze(x.numpy()))
            mask_handle = sitk.Cast(sitk.GetImageFromArray(np.squeeze(np.argmax(y.numpy(), axis=-1))), sitk.sitkFloat32)
            sitk.WriteImage(image_handle, os.path.join(base_data_path, '..', "Data", "Image.nii.gz"))
            sitk.WriteImage(mask_handle, os.path.join(base_data_path, '..', "Data", "Mask.nii.gz"))
            print(_)
    x = 1
    stop = time.time()
    print(stop - start)


if __name__ == '__main__':
    assert os.path.exists(base_data_path), "Data path not found!"
    main()
    pass
