# dir
data_root_dir = "/home/637/data/Klaus/Baby_Sleep_Awake_dataset"

# 交叉验证
num_k = 5
cross_validation_dir =  f"{data_root_dir}/CrossValidation"

# 标签设置
sleep_label = 0
awake_label = 1

# Train & Setting
batch_size = 64
epoch = 100
learning_rate = 0.0001
weight_decay = 0.001

T = 0.07
supCon_loss = 0.3
consist_loss = 0.5

