import os


def return_file_paths():
    tf_records_path = os.path.join(os.path.join('.', 'Data', 'TFRecords'))
    dicom_data_path = os.path.join('.', 'Data', 'DICOM')
    tensorboard_path = os.path.join('.', 'Data', 'Tensorboard', 'Session1')
    for d in [tf_records_path, dicom_data_path, tensorboard_path]:
        if not os.path.exists(d):
            os.makedirs(d)
    return dicom_data_path, tf_records_path, tensorboard_path

if __name__ == '__main__':
    pass
