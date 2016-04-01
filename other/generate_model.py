from facerecognition.facerecognitionmodel import FaceRecognitionModel


model = FaceRecognitionModel(dataset_directory="databases/images/fei",
                             serial_directory="databases/serialised/fei")
model.train_dataset("databases/images/fei")
model.save()
