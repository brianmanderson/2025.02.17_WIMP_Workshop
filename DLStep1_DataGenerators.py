import numpy as np
import os
from TFDataSets.TFGenerator import DataGeneratorClass
from TFDataSets import ConstantProcessors as CProcessors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import SimpleITK as sitk


def return_generator(records_path, batch=8):
    generator = DataGeneratorClass(record_paths=records_path,
                                   debug=False, repeat=True,
                                   shuffle=True,
                                   in_parallel=2, delete_old_cache=True)
    mean_value = -35
    standard_deviation_value = 60
    image_shape = (64, 256, 256)
    out_shape = (32, 128, 128)
    image_keys = ('ct_array',)
    out_keys = ('mask_array',)
    pull_keys = image_keys
    base_processors = [
        CProcessors.ExpandDimension(axis=-1, image_keys=image_keys + out_keys),
        CProcessors.DefineShape(keys=('ct_array', 'mask_array'), image_shapes=(image_shape + (1,),
                                                                        image_shape + (1,)))
    ]
    base_processors += [
        CProcessors.Cast_Data(keys=('mask_array', ), dtypes=('float32',)),
        CProcessors.RandomCrop(keys_to_crop=('ct_array', 'mask_array'),
                               crop_dimensions=out_shape + (2,),
                               min_start_stop=None), #
        CProcessors.Cast_Data(keys=('mask_array',), dtypes=('int32',)),
        CProcessors.Add_Constant(keys=image_keys,
                                 values=(-mean_value,)),  # Put them on a scale of roughly 0
        CProcessors.MultiplyImagesByConstant(keys=image_keys,
                                             values=(1/standard_deviation_value,)),
        CProcessors.Threshold_Images(keys=image_keys,
                                     lower_bounds=(-5,),
                                     upper_bounds=(5,),
                                     divides=(False,)),
        CProcessors.ToCategorical(annotation_keys=out_keys, number_of_classes=(2,)),
    ]
    base_processors += [CProcessors.ReturnOutputs(input_keys=pull_keys, output_keys=out_keys, as_tuple=False),]
    prefetch = 5
    base_processors += [{'batch': batch}, {'prefetch': prefetch}, ]
    generator.compile_data_set(image_processors=base_processors, debug=False)
    return generator


def main():
    print("Getting records for train")
    records_path = [os.path.join(os.path.join('.', 'Data', 'TFRecords'))]
    train_generator = return_generator(records_path, batch=1)
    train_dataset = train_generator.data_set
    iterator = iter(train_dataset)
    for _ in range(len(train_generator)):
        # print(_)
        x, y = next(iterator)
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(x.numpy())), os.path.join('.', 'Data', 'TFRecords', 'Image.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(y[..., 1].numpy()).astype('float')), os.path.join('.', 'Data', 'TFRecords', 'Mask.nii.gz'))
        break


if __name__ == '__main__':
    main()
    pass
