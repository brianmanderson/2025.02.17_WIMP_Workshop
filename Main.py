import os


def main():
    data_path = os.path.join('.', 'Data')
    convert_dicom_to_nifti = True
    if convert_dicom_to_nifti:
        """
        Lets convert our DICOM images and masks to two .nii files!
        """
        from Step1_DICOMToNifti import dicom_to_nifti
        print('Converting DICOM to Nifti')
        dicom_to_nifti(os.path.join(data_path, 'DICOM'))
        print('Finished!')
    create_tf_records = True
    if create_tf_records:
        """
        Lets take what we have and make some TFRecords
        """
        from Step2_MakeTFRecords import make_records
        make_records(os.path.join(data_path, 'DICOM'), out_path=os.path.join(data_path, 'TFRecords'), rewrite=True)


if __name__ == '__main__':
    main()
