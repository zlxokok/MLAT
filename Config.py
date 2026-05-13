import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = 5000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
# task_name = 'Dataset_BUSI'
task_name = "MosMes"
# task_name = 'QaTa_COV19'
learning_rate = 1e-3  # MosMeDataPlus: 1e-3, Covid19: 3e-4
batch_size = 4

model_name = 'MLAT'

train_dataset = r'C:\Users\zlx\Desktop\dataset\\' + task_name + r'\train\\'
val_dataset = r'C:\Users\zlx\Desktop\dataset\\' + task_name + r'\test\\'
test_dataset = r'C:\Users\zlx\Desktop\dataset\\' + task_name + r'\test\\'
task_dataset = r'C:\Users\zlx\Desktop\dataset\\' + task_name + r'\train\\'

# train_dataset = r'C:\Users\zlx\Desktop\dataset\Qata\QaTa-COV19\QaTa-COV19-v2\Train Set\\'
# val_dataset = r'C:\Users\zlx\Desktop\dataset\Qata\QaTa-COV19\QaTa-COV19-v2\Test Set\\'
# test_dataset = r'C:\Users\zlx\Desktop\dataset\Qata\QaTa-COV19\QaTa-COV19-v2\Test Set\\'
# task_dataset = r'C:\Users\zlx\Desktop\dataset\Qata\QaTa-COV19\QaTa-COV19-v2\Train Set\\'


session_name = r'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + r'\\' + model_name + r'\\' + session_name + r'\\'
save_path = r'C:\Users\zlx\Desktop\code\MLAT\result\MosMes' + r'\\'
model_path = save_path + r'jianfa\\'
tensorboard_folder = save_path + r'tensorboard_logs\\'
logger_path = save_path + session_name + "jianfa.log"
visualize_path = save_path + r'jianfa\\'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config

# used in testing phase, copy the session name in training phase
test_session = "Test_session_01.20_19h33jianfa"  # dice=79.98, IoU=66.83