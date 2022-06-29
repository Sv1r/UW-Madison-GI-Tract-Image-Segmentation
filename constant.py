DATAFRAME = 'df.csv'
# DATAFRAME = 'df_full.csv'
# IMAGES_PATH = 'D:\\UW_Madison_GI_Tract_Image_Segmentation\\data_2.5D\\images\\images'
# MASKS_PATH = 'D:\\UW_Madison_GI_Tract_Image_Segmentation\\data_2.5D\\masks\\masks'
IMAGE_SIZE = 256
RANDOM_STATE = 42
NUMBER_OF_CLASSES = 3
CLASSES_LIST = list(range(NUMBER_OF_CLASSES + 1))
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
EPOCH = 10
THRESHOLD = .5
K_FOLDS = 5
NUMBER_OF_INPUT_CHANNELS = 3
# Global MEAN/STD
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
