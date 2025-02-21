import os
import numpy as np
from Tools import return_file_paths
from DLStep2_MakeModel import return_model
from DLStep1_DataGenerators import return_generator, sitk

def main():
    _, data_path, tensorboard_folder = return_file_paths()
    session_id = 1
    checkpoint_path = os.path.join(tensorboard_folder, "checkpoint", f"Session_{session_id}/cp.weights.h5")

    model = return_model()
    model.load_weights(checkpoint_path)

    prediction_path = os.path.join('.', 'Data', 'Predictions')
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    train_generator = return_generator([data_path], batch=1, out_shape=(64, 256, 256))
    iterator = iter(train_generator.data_set)
    for _ in range(len(train_generator)):
        print(_)
        x, y = next(iterator)
        pred = model.predict(x)
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(x.numpy())),
                        os.path.join(prediction_path, f'{_}_Image.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(y[..., 1].numpy()).astype('float')),
                        os.path.join(prediction_path, f'{_}_Mask.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(pred[..., 1])),
                        os.path.join(prediction_path, f'{_}_Prediction.nii.gz'))


if __name__ == '__main__':
    main()
