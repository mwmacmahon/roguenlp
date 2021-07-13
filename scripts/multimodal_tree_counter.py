# %%



#%% 

# SCRATCH SCRIPT - Multi-Modal tree counter
# Shows proof-of-concept for training a network to look at 
# big trees vs small trees vs all trees vs none-of-the-above
# using both a grid intput and user text. Isn't particularly
# effective or well-organized, but I suspect that's in large part due
# to its small and incredibly crude training corpus.

# The goal is to spring off this one into something more general, 
# especially once better dataset generation has been created.
# For example:
# 1. Output list of commands and arguments via RNN/transformer decoder, 
#    potentially making some command functions trivial
#     (e.g. instead of doing ("COUNT", "BIG_TREES"), it could 
#      just put ("PRINT", "2") since it's within the algorithm's
#      capacity to solve if you feed CNN output into an output RNN)
# 2. Try something like "peek around corner" or "move to wall" and implement them
#     ("find closest wall" would be a good one, as function 
#      could then pick a good one from heatmap)
#     ("peek around corner" is another good one to test complex 
#      language understanding combined w/ map - need to understand
#      it's a movement action and what terrain it's looking at)

# This needs to be broken up into a bunch of scripts, e.g.
# 1. Dataset generation script(s)
# 1.1. Possibly seperate easy-to-run dataset generation command/script
# 1.2. Alternatively a data-generator-backend tf.data.Dataset or something like that.
# 2. Model definition script(s)
# 2.1. Since we might be playing around with structure a lot, could consider a config file/dict
# 2.2. Possibly define main chunks in one file and let user compose them in their own script?
# 3. Model training script(s)
# 4. Hyperparameter optimization script(s), if different from the training one.

# Still working out best practices on this stuff for small vs large projects,
# so all that is subject to change...

# %%

# ******************************************************************
# ************************** IMPORTANT *****************************
# ******************************************************************
# NOTICE UP FRONT REGARDING DATA REQUIREMENT: 
# Right now this script depends on a dataset I downloaded form
# https://huggingface.co/transformers/custom_datasets.html
# It can be replaced with __literally any batch of random sentences__ - 
# 
# To use the script in its current state, you need Large Movie Review Dataset
# which can be downloaded with the following commands (from that link):
#     wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#     tar -xf aclImdb_v1.tar.gz
#
# You need the "aclImdb" folder in your working directory when you run this
# script.
#
# In the future if we use a data source like this, it should be downloaded
# as part of running it if it doesn't exist. The repository's 
# /data/cache folder is .gi"tignore'd so it's the ideal place to put all 
# datasets. Alternatively, there's /tmp folders for linux, but I want to be
# cross-platform friendly...

# SUGGESTION: 
# try:
# import nltk
# nltk.download('brown')
# text = nltk.Text(nltk.corpus.brown.words()) 
# text.generate(text_length) # <- randomize amount in vaguely sentence length range
#
# There are probably better ways, too...if nothing else, download dataset in a better way!
#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras


#%%

def plot_grid(grid, hide_zeros=True):
    nrows, ncols = grid.shape
    def get_grid_char(grid_number):
        if hide_zeros and grid_number == 0:
            return "."
        else:
            return str(grid_number)

    for col_ind in range(ncols):

        row_string = "".join(get_grid_char(x) for x in grid[:,col_ind])
        print(row_string)


#%%

RANDOM_SEED = None
NUM_GRIDS = 10000 # Now it's just sampled (w/replacement), so keep that in mind
GRID_SIZE = 24
NUM_GRID_CHANNELS = 3  # starting at -1
MAX_SMALL_TREES_PER_MAP = 12
MAX_BIG_TREES_PER_MAP = 6
NUM_FEATURES = 5  # player x/y, num_big_trees, num_small_trees, num_all_trees

if RANDOM_SEED is None:
    RANDOM_SEED = np.random.randint(low=0, high=10000)

rng = np.random.default_rng(seed=RANDOM_SEED)

#%%
# Create random grids to use for training
default_grid_layout = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.int32)
default_heatmap_layout = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32)
input_grids = []
output_small_tree_heatmaps = []
output_big_tree_heatmaps = []
output_all_tree_heatmaps = []
input_grid_feature_dict = {
    # "input_text": [],
    # "output_command": [],
    "player_x": [],
    "player_y": [],
    "num_small_trees": [],
    "num_big_trees": [],
    "num_all_trees": [],
}
input_grid_features = np.zeros((NUM_GRIDS, NUM_FEATURES))
for grid_ind in range(NUM_GRIDS):
    grid = default_grid_layout.copy()

    small_tree_heatmap = default_heatmap_layout.copy()
    big_tree_heatmap = default_heatmap_layout.copy()
    all_tree_heatmap = default_heatmap_layout.copy()

    # get tree counts
    num_small_trees = rng.integers(low=0, high=MAX_SMALL_TREES_PER_MAP+1, size=1)[0]
    num_big_trees = rng.integers(low=0, high=MAX_BIG_TREES_PER_MAP+1, size=1)[0]
    num_all_trees = num_small_trees + num_big_trees

    # thin trees
    for i in range(num_small_trees):
        tree_x, tree_y = rng.integers(low=0, high=GRID_SIZE-1, size=2, dtype=np.int32)
        grid[tree_x, tree_y] = 1
        small_tree_heatmap[tree_x, tree_y] = 1
        all_tree_heatmap[tree_x, tree_y] = 1

    # thick trees
    for i in range(num_big_trees):
        tree_x, tree_y = rng.integers(low=0, high=GRID_SIZE-1, size=2, dtype=np.int32)
        grid[tree_x:tree_x+2, tree_y:tree_y+2] = np.ones((2,2), dtype=np.int32)
        big_tree_heatmap[tree_x:tree_x+2, tree_y:tree_y+2] = np.ones((2,2), dtype=np.float32)
        all_tree_heatmap[tree_x:tree_x+2, tree_y:tree_y+2] = np.ones((2,2), dtype=np.float32)

    # player location
    player_x, player_y = rng.integers(low=0, high=GRID_SIZE-1, size=2, dtype=np.int32)
    grid[player_x, player_y] = -1
    
    # Append grid + features to lists
    input_grids.append(grid)
    output_small_tree_heatmaps.append(small_tree_heatmap)
    output_big_tree_heatmaps.append(big_tree_heatmap)
    output_all_tree_heatmaps.append(all_tree_heatmap)
    # input_grid_feature_dict["input_text"].append("")
    # input_grid_feature_dict["output_command"].append("COUNT")
    input_grid_feature_dict["player_x"].append(player_x)
    input_grid_feature_dict["player_y"].append(player_y)
    input_grid_feature_dict["num_small_trees"].append(num_small_trees)
    input_grid_feature_dict["num_big_trees"].append(num_big_trees)
    input_grid_feature_dict["num_all_trees"].append(num_all_trees)
    input_grid_features[grid_ind, 0] = player_x
    input_grid_features[grid_ind, 1] = player_y
    input_grid_features[grid_ind, 2] = num_small_trees
    input_grid_features[grid_ind, 3] = num_big_trees
    input_grid_features[grid_ind, 4] = num_all_trees

for grid_ind in range(min(2,NUM_GRIDS)):
    print("------------------------------------")
    print(f"Number of small trees: {input_grid_feature_dict['num_small_trees'][grid_ind]}")
    print(f"Number of big trees: {input_grid_feature_dict['num_big_trees'][grid_ind]}")
    print(f"Player location: {(input_grid_feature_dict['player_x'][grid_ind], input_grid_feature_dict['player_y'][grid_ind])}")
    plot_grid(input_grids[grid_ind])
print(input_grid_features[:2,:])
print("Limited output to 2 maps")


# %%
# Remix the grids/heatmaps into a training set 
# Generate trivial training examples for question answering from map

beginnings1 = [
    "How many",
    "Count how many",
]
options1 = [
    " are",
    " there are",
    " are there",
    ""
]
endings1 = [
    " in the picture",
    " on the map",
    " in the picture",
    " on the map",
    ""
]
beginnings2 = [
    "Count the",
    "List the",
    "Give me the number of",
    "What is the number of",
    "Tell me the number of"
]
options2 = [
    " shown",
    " found",
    " visible",
    ""
]
endings2 = [
    " on the map",
    " in the picture",
    ""
]

objects_to_query = [
    " trees",  
    " big trees",
    " small trees",
]

# Sample grids
num_text_repeats = 40

training_input_grids = []
training_input_texts = []
training_output_labels = []
training_output_heatmaps = []
for _ in range(num_text_repeats):
    for object in objects_to_query:
        for start in beginnings1:
            for option in options1:
                for end in endings1:
                    grid_ind = rng.integers(NUM_GRIDS)  # pick random grid
                    training_input_texts.append("".join([start, object, option, end]))
                    training_input_grids.append(input_grids[grid_ind])
                    if object == " trees":
                        training_output_heatmaps.append(output_all_tree_heatmaps[grid_ind])
                        training_output_labels.append("COUNT TREE")
                    elif object == " big trees":
                        training_output_heatmaps.append(output_big_tree_heatmaps[grid_ind])
                        training_output_labels.append("COUNT BIG_TREE")
                    elif object == " small trees":
                        training_output_heatmaps.append(output_small_tree_heatmaps[grid_ind])
                        training_output_labels.append("COUNT SMALL_TREE")
                    else:
                        training_output_heatmaps.append(np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32))
                        training_output_labels.append("UNKNOWN")

        for start in beginnings2:
            for option in options2:
                for end in endings2:
                    grid_ind = rng.integers(NUM_GRIDS)  # pick random grid
                    training_input_texts.append("".join([start, object, option, end]))
                    training_input_grids.append(input_grids[grid_ind])
                    if object == " trees":
                        training_output_heatmaps.append(output_all_tree_heatmaps[grid_ind])
                        training_output_labels.append("COUNT TREE")
                    elif object == " big trees":
                        training_output_heatmaps.append(output_big_tree_heatmaps[grid_ind])
                        training_output_labels.append("COUNT BIG_TREE")
                    elif object == " small trees":
                        training_output_heatmaps.append(output_small_tree_heatmaps[grid_ind])
                        training_output_labels.append("COUNT SMALL_TREE")
                    else:
                        training_output_heatmaps.append(np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32))
                        training_output_labels.append("UNKNOWN")

#%%

# Make a tokenize function 
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertConfig
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# max_input_length = tokenizer.model_max_length  # TODO: shorten this to a much smaller max length
max_input_length = 64  # we don't need many

# Tokenization for training batches.
# For prediction instead use:
#    tokenizer.encode(input_sentence, return_tensors="tf")
def tokenize(inputs):
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenizer(  # oh hell, we do the conversion manually later for better or worse
        inputs, 
        max_length=max_input_length,
        padding="max_length",  # let's see if pad_to_multiple_of is good enough
        truncation=True, 
        return_tensors="tf"
    )

#%%
# Now start with negatives - non-meaningful phrases
# Using some code from https://huggingface.co/transformers/custom_datasets.html
# as a source of random sentences since I was messing with it elsewhere...
# ...literally any source of random sentences will do.
from pathlib import Path
import nltk
nltk.download('punkt')

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            text = text_file.read_text()[:200]  # cut it off if it's really long
            text = nltk.sent_tokenize(text)[0]  # grab first sentence
            if text.find("<br") >= 0:  # look for break html tag
                test = text.split("<br")[0]
            texts.append(text)
            labels.append("UNKNOWN")

    return texts, labels


negative_train_texts, negative_train_labels = read_imdb_split('aclImdb/train')

# Cull to same # as training examples so we can copy the input 
# We shuffle via numpy and convert back into a list...awk but it works
n_samples = len(training_input_texts)
negative_example_indices = rng.permutation(len(negative_train_texts))[:n_samples]
negative_train_texts = np.array(negative_train_texts)[negative_example_indices].tolist()
negative_train_labels = np.array(negative_train_labels)[negative_example_indices].tolist()
negative_grids = [
    input_grids[ind]
    for ind in rng.integers(0, NUM_GRIDS, n_samples)  # sample n_samples more grids
]
negative_heatmaps = [
    np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float32)
    for _ in range(n_samples)
]

# %%

# Combine normal and negative examples
training_input_texts += negative_train_texts
training_output_labels += negative_train_labels
training_input_grids += negative_grids
training_output_heatmaps += negative_heatmaps

#%%

from sklearn.preprocessing import LabelBinarizer

enc = LabelBinarizer()
training_output_labels_onehot = enc.fit_transform(training_output_labels)


#%% Separate out input grid into channels (should we do this with the ordinal encoder?)

# Convert to player and tree layers (grass is free parameter)
training_input_grid_tensors = []
for grid in training_input_grids:
    grid_tensor = np.zeros(grid.shape + tuple([NUM_GRID_CHANNELS]))
    for ind in range(NUM_GRID_CHANNELS):
        grid_tensor[:,:,ind] = (grid == ind-1).astype(np.float32)
    training_input_grid_tensors.append(grid_tensor)

# %%
# Split train into train and val, leaving test alone
training_set_size = int(np.round(0.9 * len(training_input_texts)))
shuffled_indices = rng.permutation(len(training_input_texts))
train_indices = shuffled_indices[:training_set_size]
val_indices = shuffled_indices[training_set_size:]


train_texts = np.array(training_input_texts)[train_indices]
train_labels = np.array(training_output_labels_onehot)[train_indices]
train_grids = np.stack(training_input_grid_tensors)[train_indices]
train_heatmaps = np.stack(training_output_heatmaps)[train_indices]

val_texts = np.array(training_input_texts)[val_indices]
val_labels = np.array(training_output_labels_onehot)[val_indices]
val_grids = np.stack(training_input_grid_tensors)[val_indices]
val_heatmaps = np.stack(training_output_heatmaps)[val_indices]


# %%
# Encode dataset
train_encodings = tokenize(train_texts.tolist())
val_encodings = tokenize(val_texts.tolist())


#%%



#%%

# train_dataset_x = tf.data.Dataset.from_tensor_slices([  # Using lists
#     train_encodings["input_ids"],
#     train_encodings["attention_mask"],
#     train_grids,
# ])
# train_dataset_y = tf.data.Dataset.from_tensor_slices([  # Using lists
#     train_labels,
#     train_heatmaps,
# ])
train_dataset_x = tf.data.Dataset.from_tensor_slices({  # Using dicts
    "input_ids": train_encodings["input_ids"],
    "input_attention_masks": train_encodings["attention_mask"],
    "input_grids": train_grids,
})
train_dataset_y = tf.data.Dataset.from_tensor_slices({  # Using dicts
    "output_labels": train_labels,
    "output_heatmaps": train_heatmaps,
})
train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))

# val_dataset_x = tf.data.Dataset.from_tensor_slices([  # Using lists
#     val_encodings["input_ids"],
#     val_encodings["attention_mask"],
#     val_grids,
# ])
# val_dataset_y = tf.data.Dataset.from_tensor_slices([  # Using lists
#     val_labels,
#     val_heatmaps,
# ])
val_dataset_x = tf.data.Dataset.from_tensor_slices({  # Using dicts
    "input_ids": val_encodings["input_ids"],
    "input_attention_masks": val_encodings["attention_mask"],
    "input_grids": val_grids,
})
val_dataset_y = tf.data.Dataset.from_tensor_slices({  # Using dicts
    "output_labels": val_labels,
    "output_heatmaps": val_heatmaps,
})
val_dataset = tf.data.Dataset.zip((val_dataset_x, val_dataset_y))

#%%

input_ids_placeholder = tf.keras.layers.Input(
    shape=(max_input_length,),  # without comma, will reduce to scalar.
    # tensor=test_input,  # hard-codes the length of THIS sequence, useless
    dtype=tf.int32,
    name="input_ids"
)
attention_mask_placeholder = tf.keras.layers.Input(
    shape=(max_input_length,),  # without comma, will reduce to scalar.
    # tensor=test_input,  # hard-codes the length of THIS sequence, useless
    dtype=tf.int32,
    name="input_attention_masks"
)
input_grid_placeholder = keras.Input(
    shape=(GRID_SIZE, GRID_SIZE, NUM_GRID_CHANNELS),
    dtype=np.float32,
    name="input_grids"
)

# %%
# Load pre-trained model, but configure it to have a dense layer on top and no decoder,
# though it was of course pre-trained with a masked LM decoder.

# Load text and run through distilbert as encoder
NUM_SENTENCE_EMBED_FEATURES = 256
config = DistilBertConfig(
    dropout=0.2,
    attention_dropout=0.2,
    num_labels=NUM_SENTENCE_EMBED_FEATURES,  #only works for sequence classifier
    pad_token_id=tokenizer.eos_token_id  # take only one sentence?
)
config.output_hidden_states = False
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(
# distilbert_model = TFDistilBertModel.from_pretrained(
    'distilbert-base-uncased',
    config=config
)

sentence_embedding = distilbert_model(
    input_ids=input_ids_placeholder,
    attention_mask=attention_mask_placeholder,
    # other distilbert options go here, e.g. outputting hidden states
).logits  # just pull out logits
# # SIMPLE DENSE TOP LAYER
# cls_token = distilbert_layer[0][:,0,:]  # we only care about start token, not arbitary # word tokens
# # ^^^ WARNING: this operation causes issues as it's not a proper Layer operation - need to subclass (lambda not recommended)
# X = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)

# Recognizing command - for now we don't use map at all (later we will have that as input too)
X2 = tf.keras.layers.BatchNormalization(momentum=0.99)(sentence_embedding)
# X2 = sentence_embedding
X2 = tf.keras.layers.Dense(NUM_SENTENCE_EMBED_FEATURES, activation='relu')(X2)
X2 = tf.keras.layers.Dropout(0.2)(X2)
X2 = tf.keras.layers.Dense(NUM_SENTENCE_EMBED_FEATURES, activation='relu')(X2)
X2 = tf.keras.layers.BatchNormalization()(X2)
X2 = tf.keras.layers.Dropout(0.2)(X2)
X2 = tf.keras.layers.Dense(NUM_SENTENCE_EMBED_FEATURES // 2, activation='relu')(X2)
X2 = tf.keras.layers.BatchNormalization()(X2)
X2 = tf.keras.layers.Dropout(0.2)(X2)
X2 = tf.keras.layers.Dense(NUM_SENTENCE_EMBED_FEATURES // 4, activation='relu')(X2)
embed_dense_block_output = tf.keras.layers.BatchNormalization()(X2)
output_labels = tf.keras.layers.Dense(4, activation='softmax', name="output_labels")(embed_dense_block_output)
# ^ use softmax activation if classification layer

# %%


# Define CNN functions
# Using:  https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
    relu = keras.layers.ReLU()(inputs)
    bn = keras.layers.BatchNormalization(momentum=0.98)(relu)
    # bn = relu
    return bn

def residual_block(
        x: tf.Tensor, 
        filters: int, 
        kernel_size: int = 3, 
        downsample: bool = False
    ) -> tf.Tensor:
    y = keras.layers.Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = keras.layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = keras.layers.Conv2D(kernel_size=1,
                                strides=2,
                                filters=filters,
                                padding="same")(x)
    # TODO: add upsample layer option w/ keras.layers.UpSample2D?
    out = keras.layers.Add()([x, y])
    out = relu_bn(out)
    return out

# This is crude but I think it should work for now
def mix_in_nlp_embedding(
    X: tf.Tensor,
    sentence_embedding: tf.Tensor,
    io_num_filters: int,
    grid_size: int,
):
    # Expects same # filters for input layer, embedding, and output
    # so downscale embedding:
    embed_1d = tf.keras.layers.Dense(io_num_filters, activation='relu')(sentence_embedding)
    # embed_1d = sentence_embedding

    # Concatenate embedding (broadcasted into 2D) with input layer, so 2x channels
    embed_2d = keras.layers.Reshape((1,1,io_num_filters))(embed_1d)
    embed_2d = keras.layers.UpSampling2D(grid_size)(embed_2d)
    X = keras.layers.Concatenate()([X, embed_2d])

    # convert back to original # filters with a little bit of horizontal context
    X = keras.layers.Conv2D(
        kernel_size=(1, 1),  # a little horizontal context
        strides=1, 
        filters=io_num_filters,
        padding="same"
    )(X)
    return X


# %%
RESIDUAL_BLOCK_N_FEATURES = 32
NUM_BLOCKS_PRE_NLP = 3
NUM_BLOCKS_POST_NLP = 3

# Now run map through CNN
X = keras.layers.BatchNormalization(momentum=0.98)(input_grid_placeholder)  # normalize inputs
# X = input_grid_placeholder
X = keras.layers.Conv2D(  # move to many channels from input features/embeddings
    RESIDUAL_BLOCK_N_FEATURES,  
    (1, 1),
    padding="same", 
    # activation='relu',  # we'll do this one manually
)(X)
X = relu_bn(X)  # activate and re-normalize

# Switch to residual blocks
# If # filters til now is not same as residual block # filters,
# you will need to add an intermediate layer
for i in range(NUM_BLOCKS_PRE_NLP):
    X = residual_block(X, filters=RESIDUAL_BLOCK_N_FEATURES, kernel_size=3)

# Concatenate sentence embedding onto every single grid element as extra channels
X = mix_in_nlp_embedding(
    X, embed_dense_block_output, #sentence_embedding,
    io_num_filters=RESIDUAL_BLOCK_N_FEATURES,
    grid_size=GRID_SIZE
)
for i in range(NUM_BLOCKS_POST_NLP):
    X = residual_block(X, filters=RESIDUAL_BLOCK_N_FEATURES, kernel_size=3)

# Get final heatmap (1 channel) with a 1x1 convolution
output_heatmaps = keras.layers.Conv2D(
    filters=1,
    kernel_size=(1,1),
    activation='relu',
    padding='same',  # don't shrink
    name="output_heatmaps"
)(X)

# # From article: could modify this for pseudo-hourglass
# num_blocks_list = [2, 5, 5, 2]
# for i in range(len(num_blocks_list)):
#     num_blocks = num_blocks_list[i]
#     for j in range(num_blocks):
#         t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
#     num_filters *= 2


# # # spatial reduction / feature increase - not needed unless we hourglass...
# # X = keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(X)
# # X = keras.layers.MaxPooling2D((2, 2))(X)
# # X = keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu')(X)
# # X = keras.layers.MaxPooling2D((2, 2))(X)




# # output_features = keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(output_features)
# # output_features = keras.layers.Flatten()(output_features)

# # x = keras.ReLU()(x)
# # x = keras.Conv2D(
# #     filters=16,
# #     kernel_size=3,
# #     strides=1,
# #     padding='same',
# #     kernel_initializer='he_normal'
# # )(input_grid_placeholder)
# # x = keras.BatchNormalization(momentum=0.9)(x)
# # x = keras.ReLU()(x)
# # x = keras.Conv2D(
# #     filters=16,
# #     kernel_size=3,
# #     strides=1,
# #     padding='same',
# #     kernel_initializer='he_normal'
# # )(input_grid_placeholder)
# # x = keras.BatchNormalization(momentum=0.9)(x)
# # x = keras.ReLU()(x)

# # # Cut size by factor of two
# # x = keras.Conv2D(  # 
# #     filters=32,
# #     kernel_size=1,
# #     strides=2,
# #     padding='same',
# #     kernel_initializer='he_normal'
# # )(x)






# # TEST ALTERNATIVE

# # Get final heatmap (1 channel) with a 1x1 convolution
# output_heatmaps = keras.layers.Conv2D(
#     filters=1,
#     kernel_size=(1,1),
#     activation='relu',
#     padding='same',  # don't shrink
#     name="output_heatmaps"
# )(input_grid_placeholder)


# %% 


# %%

# inputs = [input_ids_placeholder, attention_mask_placeholder, input_grid_placeholder]  # Using lists
# outputs = [output_labels, output_heatmaps]  # Using lists
inputs = {  # Using dicts
    "input_ids": input_ids_placeholder,
    "input_attention_masks": attention_mask_placeholder,
    "input_grids": input_grid_placeholder
}
outputs = {  # Using dicts
    "output_labels": output_labels,  # raw cnn output (unused atm)
    "output_heatmaps": output_heatmaps,  # coords
}
model = tf.keras.Model(inputs, outputs)
model.summary()

# %%

# Test model actually takes input data
# model([  # Using lists
#     train_encodings["input_ids"][-5:],
#     train_encodings["attention_mask"][-5:],
#     train_grids[-5:]
# ])
model({  # Using dicts
    "input_ids": train_encodings["input_ids"][-5:],
    "input_attention_masks": train_encodings["attention_mask"][-5:],
    "input_grids": train_grids[-5:]
})

# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(
    optimizer=optimizer,
    # loss=tf.keras.losses.MeanSquaredError(),
    loss={
        "output_labels": tf.keras.losses.CategoricalCrossentropy(
            from_logits=False  # otherwise assumes normalized inputs, but we used softmax
            ),  # classification of user input type
        # "output_labels": tf.keras.losses.MeanSquaredError(),  # try this instead?
        "output_heatmaps": tf.keras.losses.MeanSquaredError(),  # coords
    },
    loss_weights={
        "output_labels": 1,
        "output_heatmaps": 1,
    },
    # metrics=['accuracy']  # caused nightmarish and completely mysterious errors
)

# %%

# Test model can _evaluate_ data without crashing. outputting actual loss, eh
# model.evaluate([  # Using lists
#     train_encodings["input_ids"][-5:],
#     train_encodings["attention_mask"][-5:],
#     train_grids[-5:]
# ])
model.evaluate({  # Using dicts
    "input_ids": train_encodings["input_ids"][-5:],
    "input_attention_masks": train_encodings["attention_mask"][-5:],
    "input_grids": train_grids[-5:]
})

# %%

history = model.fit(
    train_dataset.shuffle(buffer_size=20000).batch(64),  # from 1024
    epochs=5, 
    # validation_split=0.2
    validation_data=val_dataset.shuffle(buffer_size=20000).batch(64)  # always says shape is wrong, even if ok on training
)

# %%

# # Skip batching, etc, we're using a training manager following the example
# # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(
#     optimizer=optimizer, 
#     loss=tf.keras.losses.MeanSquaredError()
# ) # any keras loss fn, or model.compute_loss if implemented
# model.fit(
#     train_dataset.shuffle(1000).batch(16), 
#     epochs=1, 
#     batch_size=16
# )

#%%

# Create some functions for testing the trained model
label_decoder_dict = {
    num: label
    for num, label in enumerate(enc.classes_)
}
label_decoder_dict

def create_input(input_str, grid_ind):
    token_obj = tokenize(input_str)
    output = {
        "input_ids": token_obj["input_ids"],
        "input_attention_masks": token_obj["attention_mask"],
        "input_grids": np.expand_dims(training_input_grid_tensors[grid_ind], 0)
    }
    print("INPUT GRID:")
    plot_grid(training_input_grids[grid_ind])
    print(f"INPUT TEXT: {input_str}")
    return output

# %%
import matplotlib.pyplot as plt

# Test results
# input_text = "How many of the gigantic trees are visible"
# input_text = "Count all the huge trees"
# input_text = "Angry crowd of llamas "
input_text = "Look at the trees"
grid_ind = rng.integers(0, len(training_input_texts))
test_input = create_input(
    input_text, grid_ind
)

test_output = model(test_input)
plt.imshow(np.array(test_output["output_heatmaps"][0,:,:,0]).transpose())
output_choice = np.argmax(test_output["output_labels"])
output_choice_confidence = 100 * test_output["output_labels"][0, output_choice]
output_choice_label = label_decoder_dict[output_choice]
print(f"OUTPUT LABEL: {output_choice_label} ({output_choice_confidence}%)")
print(f"All output probabilities: {test_output['output_labels']}")
# %%

# test_output["output_labels"]
# train_labels.sum(axis=0)/train_labels.sum()



# %% 


# import regex as re
# def split_sentences(input_string):
    # max_num_splits = 10
    # return ''.join(
    #     sentence + '.'
    #     for sentence in re.split(
    #         '\.(?=\s*(?:[A-Z]|$))',
    #         input_string,
    #         maxsplit=max_num_splits
    #     )[:-1]
    # )

# split_sentences("what are you talking about? oh, really.")[0]
# %% 


# %% 

# val_encodings.sum(axis=1)

# %% 

# a = next(iter(val_dataset.batch(2)))[1]#["output_labels"]
# b = next(iter(val_dataset.batch(2)))[0]
# b_dict = {
    
#     "input_ids": b["input_ids"],
#     "input_attention_masks": b["input_attention_masks"],
#     "input_grids": b["input_grids"]
# }
# c = model(b_dict)#["output_labels"]  # works
# tf.keras.losses.CategoricalCrossentropy()(a["output_labels"], c["output_labels"]) # Works
# # %%
# model.evaluate(b_dict, a)  # FAILS! why!
# # %%
