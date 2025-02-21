from Tools import return_file_paths


def main():
    dicom_data_path, tf_records_path, tensorboard_path = return_file_paths()
    convert_dicom_to_nifti = True
    if convert_dicom_to_nifti:
        """
        Lets convert our DICOM images and masks to two .nii files!
        """
        from Step1_DICOMToNifti import dicom_to_nifti
        print('Converting DICOM to Nifti')
        dicom_to_nifti(dicom_data_path)
        print('Finished!')
    create_tf_records = True
    if create_tf_records:
        """
        Lets take what we have and make some TFRecords
        """
        from Step2_MakeTFRecords import make_records
        make_records(dicom_data_path, out_path=tf_records_path, rewrite=True)


if __name__ == '__main__':
    main()
