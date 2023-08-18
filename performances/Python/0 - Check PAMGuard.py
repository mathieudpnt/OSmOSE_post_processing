from post_processing_detections.utilities.def_func import find_files

wav_files = find_files(f_type='file', ext='wav')
pgdf_files = find_files(f_type='dir', ext='pgdf')


print('\n{0} wav files selected'.format(len(wav_files)))
print('{0} pgdf files selected'.format(len(pgdf_files)))
