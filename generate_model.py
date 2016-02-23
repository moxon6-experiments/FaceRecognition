from facerecognition.featuremodel import FeatureModel

model = FeatureModel("data/fei_database")
model.train_dataset("data/fei_database")
model.save("fei_serialised")
