import os
from os import listdir
import pandas as pd
import numpy as np
import random
import datetime
import tensorflow as tf
import tensorflow

#color
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121

from data_generator_single_v10082020 import DataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tensorflow.keras.backend.clear_session()
from tensorflow.keras.callbacks import CSVLogger

physical_devices = tensorflow.config.list_physical_devices('GPU')

#### MULTI INPUT MODEL

# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
#   tf.config.experimental.set_memory_growth(physical_devices[1], True)
#
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass


# Attain imags tat are actually processed
valid_img_id = os.listdir('../../data_m21t/train_npy_224_v10032020/')
valid_img_id = [el[0:-4] for el in valid_img_id]

# Load training dataframe
train_df = pd.read_csv('../data/csv_files/train_v10082020.csv')
train_df = train_df[train_df['SOPInstanceUID'].isin(valid_img_id)]



# we are partitioning the data by the StudyInstanceUID and stratifying by negative_exam_for_pe
train_df['label_v10082020'] = train_df['label_v10082020'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [int(el) for el in x])
Xy_df = train_df[['StudyInstanceUID', 'label_v10082020', 'pe_present_on_image']]

# X is the label of the image and y is the binary label regarding negative_exam
X = Xy_df['StudyInstanceUID'].tolist()
y = Xy_df['label_v10082020'].tolist()
strat = Xy_df['pe_present_on_image'].tolist()

# split
train_id_list, valid_id_list, _, _ = train_test_split(X, y, test_size = 0.33, stratify = strat)

train_id_list = train_df[(train_df['StudyInstanceUID'].isin(train_id_list)) & (train_df['SOPInstanceUID'].isin(valid_img_id)) ]['SOPInstanceUID'].unique().tolist()
valid_id_list = train_df[(train_df['StudyInstanceUID'].isin(valid_id_list)) & (train_df['SOPInstanceUID'].isin(valid_img_id)) ]['SOPInstanceUID'].unique().tolist()


# creating labels (what we are predicting)
CLASSES = ['rightsided_pe', 'leftsided_pe', 'central_pe',
             'acute_pe', 'chronic_pe', 'qa_motion', 'qa_contrast',
             'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']

# train_df['label'] = train_df.apply(lambda x: x[CLASSES].values.tolist(), axis = 1)'

# create partition
partition = {
    'train': train_id_list,
    'validation': valid_id_list
}


labels = dict(zip(train_df['SOPInstanceUID'], train_df['label_v10082020']))

BATCH_SIZE = 64
ST_PR_EP_TR = int(len(train_id_list) // BATCH_SIZE) # total train count / batch size
ST_PR_EP_VA = int(len(valid_id_list) // BATCH_SIZE) # total valid count / batch


# Parameters
# Parameters
params = {'dim': (224,224),
          'batch_size': BATCH_SIZE,
          'n_classes': len(CLASSES),
          'n_channels': 3,
          'data_path': '../../data_m21t/train_npy_224_v10032020/',
          'shuffle': True}


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
training_generator_tf = tf.data.Dataset.from_generator(
    training_generator,
    output_types=(tf.float32, tf.float32)
)
training_generator_tf = training_generator_tf.prefetch(tf.contrib.data.AUTOTUNE)

validation_generator = DataGenerator(partition['validation'], labels, **params)
validation_generator_tf = tf.data.Dataset.from_generator(
    validation_generator,
    output_types=(tf.float32, tf.float32)
)
validation_generator_tf = validation_generator_tf.prefetch(tf.contrib.data.AUTOTUNE)

# strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:1"])
# with strategy.scope():
    # 1. load pretrained model

baseModel = DenseNet121(weights='imagenet', include_top = False, input_shape = (224, 224, 3))

# add additional layers for the current task
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="tanh")(headModel)
headModel = Dropout(0.5)(headModel)

# multiple heads\
rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(headModel)
lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(headModel)
cpe = Dense(1, activation='sigmoid', name='central_pe')(headModel)
acpe = Dense(1, activation='sigmoid', name='acute_pe')(headModel)
cnpe = Dense(1, activation='sigmoid', name='chronic_pe')(headModel)
qamo = Dense(1, activation='sigmoid', name='qa_motion')(headModel)
qaco = Dense(1, activation='sigmoid', name='qa_contrast')(headModel)
rv_gt1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(headModel)
rv_lt1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(headModel)

# 2. freeze layers with weights
baseModel.trainable = False
auc = tensorflow.keras.metrics.AUC(name = 'auc')
model = Model(inputs=baseModel.input, outputs=[rspe, lspe, cpe, acpe, cnpe, qamo, qaco, rv_gt1, rv_lt1])

# 09-28-2020: try adadelta optimizer and early stopping
optimizer = tensorflow.keras.optimizers.Adadelta()
earlystop_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience = 3)

# ref: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
losses = {
	"rightsided_pe": tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"leftsided_pe": tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"central_pe":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"acute_pe":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"chronic_pe":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"qa_motion":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"qa_contrast":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"rv_lv_ratio_gte_1":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
	"rv_lv_ratio_lt_1":tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
}

loss_weights = {
"rightsided_pe":0.01,
"leftsided_pe":0.01,
"central_pe":0.01,
"acute_pe":0.01,
"chronic_pe":0.01,
"rv_lv_ratio_gte_1":0.01,
"rv_lv_ratio_lt_1":0.01,
    "qa_motion": 0.015,
	"qa_contrast": 0.015,
}

load_checkpoint_path = 'training_10082020_single_input/cp.ckpt'
# model.load_weights(load_checkpoint_path)

model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights = loss_weights,
    metrics = [auc],
    # weighted_metrics = ['negative_exam_for_pe','pe_present_on_image','rv_lv_ratio_gte_1','rv_lv_ratio_lt_1','leftsided_pe','chronic_pe','rightsided_pe','acute_and_chronic_pe','central_pe','indeterminate']
    )

# https://www.tensorflow.org/tensorboard/scalars_and_keras
import datetime
logdir = "logs/scalars/single_input_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)

# adding class weights
# ref: https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
# in case with multiple outputs
# ref: https://github.com/keras-team/keras/issues/11735
class_weight_dict = {
    'negative_exam_for_pe': {0: 579449, 1: 1211145},
    'pe_present_on_image': {0: 96540, 1: 1694054},
    'rv_lv_ratio_gte_1': {0: 230539, 1: 1560055},
    'rv_lv_ratio_lt_1': {0: 312230, 1: 1478364},
    'leftsided_pe': {0: 377634, 1: 1412960},
    'chronic_pe': {0: 71874, 1: 1718720},
    'rightsided_pe': {0: 461195, 1: 1329399},
    'acute_and_chronic_pe': {0: 34842, 1: 1755752},
    'central_pe': {0: 97531, 1: 1693063},
    'indeterminate': {0: 36680, 1: 1753914}
        }

save_checkpoint_path = "training_10082020/10082020_single_input.ckpt"
checkpoint_dir = os.path.dirname(save_checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=save_checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

csv_logger_callback = CSVLogger("model_history_log_v10082020_single_input.csv", append=True)


model.fit(x=training_generator_tf,
                    validation_data=validation_generator_tf,
                    use_multiprocessing=False,
                    workers = 16,
                    steps_per_epoch = ST_PR_EP_TR,
                    validation_steps = ST_PR_EP_VA,
                    epochs = 2000,
                    callbacks = [tensorboard_callback, earlystop_callback, cp_callback, csv_logger_callback]
                    # verbose = 0
                    # class_weight = class_weight_dict
                    )

print('Model Saved!')
model.save_weights('./checkpoints/my_checkpoint')
