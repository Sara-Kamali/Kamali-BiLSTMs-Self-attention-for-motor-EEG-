# Entere the required libraries
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, Layer,  LayerNormalization
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

# Import data that is saved in .mat format
# Setup and Save Directory
save_dir = r"C:\Users\..."
os.makedirs(save_dir, exist_ok=True)

#Load Data
mat_file = r"C:\Users\..."
mat_data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

# Load global time vector (in ms) for the selected window of downsampled data
time_vec_ds = mat_data['time_vec_window_ds'].squeeze()

# Downsampled EEG data of brain areas,  data format is (num_subs*num-trls, num_samples, 3), where 3 is the number of bands: alpha, beta, low gamma
eeg_mat_ds = mat_data['all_sp_norm_ds'] 


# Label of the EEG data for brain areas, the format is (num_subs*num-trls, num_samples)
label_mat_ds = mat_data['label_data_ds']

left_eeg_ds = eeg_mat_ds[1] # left sensorimotor EEG data
right_eeg_ds = eeg_mat_ds[3] # right sensorimotor EEG data
left_labels_ds = label_mat_ds[1] # left sensorimotor eeg 
right_labels_ds = label_mat_ds[3] # right sensorimotor eeg
left_sub_id = sub_id[1]
right_sub_id = sub_id[3]
# Subjects' IDs in the form of (1, num_trials)
sub_id = mat_data['subID_data']

# Select LSM/RSM one per time
subject_trials = left_eeg_ds[:,:,:-1] # shape: (num_subs*num-trls, 256, 2): We only classify based on mu and beta bands
subject_labels = left_labels_ds      # shape: (num_subs*num-trls, 256)
subject_labels = subject_labels.astype(np.int32)  # ensure integer labels
time_vec_ds = time_vec_ds[1] - 2000 # Go cue is the time reference
print(subject_trials.shape)


# Function to plot loos and accuracy per epoch toobserve against overfitting
# Plot Loss & Accuracy per Subject
def plot_loss(history, subject_idx, save_dir):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(10, 4))
    # --- Loss ---
    plt.subplot(1, 2, 1)
    if 'loss' in history.history:
        plt.plot(epochs, history.history['loss'], label='Train Loss', 
                 color='black', marker='o', linestyle='--')
    if 'val_loss' in history.history:
      plt.plot(epochs, history.history['val_loss'], label='Val Loss', 
               color='purple', marker='o', linestyle='--')
    plt.xticks(ticks=np.arange(1, len(epochs) + 1, 2))
    plt.title(f"Loss - Subject {subject_idx+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # --- Accuracy ---
    plt.subplot(1, 2, 2)
    if 'sparse_categorical_accuracy' in history.history:
        plt.plot(epochs, history.history['sparse_categorical_accuracy'], 
                 label='Train Acc', color='black', marker='o', linestyle='--')
    if 'val_sparse_categorical_accuracy' in history.history:
        plt.plot(epochs, history.history['val_sparse_categorical_accuracy'], 
                 label='Val Acc', color='purple', marker='o', linestyle='--')
    plt.xticks(ticks=np.arange(1, len(epochs) + 1, 2))
    plt.title(f"Accuracy - Subject {subject_idx+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_acc_subject_{subject_idx+1}.png"), dpi=300)
    plt.show()


## The model
#Network architecture is two BiLSTM layers + self-atention layer, dense (softmax) output layer, with dropout layers in between
# Self-Attention Layer (single-head) with residual connection and projection
class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, key_dim, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        # Dense layers to compute queries, keys, and values
        self.query_dense = Dense(key_dim)
        self.key_dense = Dense(key_dim)
        self.value_dense = Dense(key_dim)
        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.proj = None  # Projection layer to match input dimension

    def build(self, input_shape):
        # Create projection layer to map attention output back to input dimension
        self.proj = Dense(input_shape[-1])
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Compute queries, keys, and values
        Q = self.query_dense(inputs)    # shape: (batch, time, key_dim)
        K = self.key_dense(inputs)        # shape: (batch, time, key_dim)
        V = self.value_dense(inputs)      # shape: (batch, time, key_dim)
        
        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, time, time)
        scores = scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attn_weights = tf.nn.softmax(scores, axis=-1)  # (batch, time, time)
        attn_output = tf.matmul(attn_weights, V)        # (batch, time, key_dim)
        attn_output = self.dropout(attn_output, training=training)
        
        # Project the attention output to match the input dimension
        proj_attn_output = self.proj(attn_output)
        context = self.norm(inputs + proj_attn_output)
        if training:
            return context
        return context, proj_attn_output, attn_weights

# Train-time model builder with LSTM backbone
def build_model_for_training(input_dim, hidden_dim, num_labels, time_steps, first_layer_dim):
    inputs = Input(shape=(time_steps, input_dim))
    x = Bidirectional(LSTM(first_layer_dim//2, return_sequences=True,
                           dropout=0.5, recurrent_dropout=0.5))(inputs)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(hidden_dim//2, return_sequences=True,
                           dropout=0.4, recurrent_dropout=0.4))(x)
    x = Dropout(0.4)(x)
    context = SelfAttentionLayer(key_dim=hidden_dim)(x)
    # When training, SelfAttentionLayer returns a single tensor.
    if isinstance(context, tuple):
        context = context[0]
    x = Dropout(0.3)(context)
    outputs = TimeDistributed(Dense(num_labels, activation='softmax'))(x)
    return Model(inputs=inputs, outputs=outputs)

# Inference-time model builder with LSTM backbone
def build_model_for_inference(input_dim, hidden_dim, num_labels, time_steps, first_layer_dim):
    inputs = Input(shape=(time_steps, input_dim))
    x = Bidirectional(LSTM(first_layer_dim//2, return_sequences=True,
                           dropout=0.5, recurrent_dropout=0.5))(inputs)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(hidden_dim//2, return_sequences=True,
                           dropout=0.4, recurrent_dropout=0.4))(x)
    x = Dropout(0.4)(x)
    context, _, attn_weights = SelfAttentionLayer(key_dim=hidden_dim)(x)
    x = Dropout(0.3)(context)
    outputs = TimeDistributed(Dense(num_labels, activation='softmax'))(x)
    return Model(inputs=inputs, outputs=[outputs, attn_weights])

# Leave-one-subject-out cross-validation loop
num_subjects = subject_trials.shape[0] // 27 # because each subject has 27 trials
all_accuracy = []
all_precision = [[] for _ in range(2)]
all_f1 = [[] for _ in range(2)]
all_recall = [[] for _ in range(2)]
all_attention_weights = []
all_class_accuracy = [[] for _ in range(2)]
overall_precision = []
overall_f1 = []
per_subject_sample_accuracy = []
# to store ROC data for each LOSO round
all_fpr = []
all_tpr = []
all_auc_list = []

for subject_idx in range(num_subjects):
    val_idx = np.arange(subject_idx * 27, (subject_idx + 1) * 27) # to seperate the trials of the current subject 
    train_idx = np.setdiff1d(np.arange(subject_trials.shape[0]), val_idx) # rest of the data is used for training/test

    X_train, y_train = subject_trials[train_idx], subject_labels[train_idx]
    X_val, y_val = subject_trials[val_idx], subject_labels[val_idx]

    # Build and train model
    model = build_model_for_training(input_dim=X_train.shape[-1], hidden_dim=hidden_dim, 
                                     num_labels=2, time_steps=X_train.shape[1], first_layer_dim=first_layer_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    # Internal validation from training set, split to 90% to 10%
    X_tr, X_val_internal, y_tr, y_val_internal = train_test_split(
    X_train, y_train, test_size=0.1, random_state=subject_idx)

    history = model.fit(X_tr, y_tr,
                        validation_data=(X_val_internal, y_val_internal),
                        epochs=20,
                        batch_size=4,
                        callbacks=[
                            EarlyStopping(patience=5, min_delta=1e-4, restore_best_weights=True),
                            ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5)],
                            verbose=0
                       )

    plot_loss(history, subject_idx, save_dir) # Observe overfitting

    # Build inference model and transfer weights
    infer_model = build_model_for_inference(input_dim=X_train.shape[-1], hidden_dim=hidden_dim, 
                                            num_labels=2, time_steps=X_val.shape[1], first_layer_dim=first_layer_dim)
    infer_model.set_weights(model.get_weights())

    # Get predictions and attention weights
    y_pred, attn_weights = infer_model.predict(X_val, verbose=0)
    # For single-head, attn_weights shape is (batch, T, T); add head dimension for consistency.
    if attn_weights.ndim == 3:
        attn_weights = np.expand_dims(attn_weights, axis=1)
    all_attention_weights.append(attn_weights.astype(np.float32))

    # Accuracy and classification metrics
    y_pred_labels = np.argmax(y_pred, axis=-1)
    sample_accuracy = np.mean((y_pred_labels == y_val), axis=0)
    per_subject_sample_accuracy.append(sample_accuracy)

    y_true_flat = y_val.flatten()
    y_pred_flat = y_pred_labels.flatten()
    y_prob_flat = y_pred.reshape(-1, y_pred.shape[-1])
    valid_mask = y_true_flat >= 0

    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]
    y_prob_flat = y_prob_flat[valid_mask]

    acc = accuracy_score(y_true_flat, y_pred_flat)
    all_accuracy.append(acc)
    print(f"\nSubject {subject_idx+1}: Accuracy={acc:.3f}")

    for cls in range(2):
        cls_mask = (y_true_flat == cls)
        if np.any(cls_mask):
            cls_prec = precision_score(y_true_flat, y_pred_flat, pos_label=cls, average='binary', zero_division=0)
            cls_f1 = f1_score(y_true_flat, y_pred_flat, pos_label=cls, average='binary', zero_division=0)
            cls_rec = recall_score(y_true_flat, y_pred_flat, pos_label=cls, average='binary', zero_division=0)
            cls_acc = accuracy_score(cls_mask, y_pred_flat == cls)
            try:
                cls_auc = roc_auc_score((y_true_flat == cls).astype(int), y_prob_flat[:, cls])
            except:
                cls_auc = 0.0
        else:
            cls_prec = cls_f1 = cls_rec = cls_acc = cls_auc = 0.0

        all_precision[cls].append(cls_prec)
        all_f1[cls].append(cls_f1)
        all_recall[cls].append(cls_rec)
        all_class_accuracy[cls].append(cls_acc)

        print(f" Class {cls} - Precision: {cls_prec:.3f}, F1: {cls_f1:.3f}, Recall: {cls_rec:.3f}, Accuracy: {cls_acc:.3f}, AUC: {cls_auc:.3f}")

    overall_prec = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    overall_f1_score = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    try:
        overall_auc = roc_auc_score(y_true_flat, y_prob_flat[:, 1])
    except:
        overall_auc = 0.0

    overall_precision.append(overall_prec)
    overall_f1.append(overall_f1_score)

    print(f" Overall - Precision: {overall_prec:.3f}, F1: {overall_f1_score:.3f}, AUC: {overall_auc:.3f}")

    # --- ROC Curve for this subject ---
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_prob_flat[:, 1])
    roc_auc = auc(fpr, tpr)
    # Store the ROC data for later averaging
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc_list.append(roc_auc)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Subject {subject_idx+1}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ROC_Curve_subject_{subject_idx+1}.png"), dpi=300)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - Subject {subject_idx+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Confusion_Matrix_subject_{subject_idx+1}.png"), dpi=300)
    # plt.close()

    # --- Sample-wise Accuracy Plot ---
    plt.figure(figsize=(8, 3))
    plt.plot(time_vec_ds, sample_accuracy, linewidth=2)
    plt.title(f"Sample-wise Accuracy Curve - Subject {subject_idx+1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Accuracy_per_sample_subject_{subject_idx+1}.png"), dpi=300)
    # plt.close()
    
    # Compute average attention over all batch samples  (batch,  T, T))
    # Average over batch→ shape: (T, T)
    attn_avg_matrix = np.mean(attn_weights, axis=(0,1))
    # For plotting a curve, we can average over the query dimension to get a single time-series: shape (T,)
    attn_avg_curve = np.mean(attn_avg_matrix, axis=0)
    attn_norm = (attn_avg_curve - np.min(attn_avg_curve)) / (np.ptp(attn_avg_curve) + 1e-8)

    # Plot both sample accuracy and normalized attention curve
    plt.figure(figsize=(10, 4))
    plt.plot(time_vec_ds, sample_accuracy, label="Sample Accuracy", linewidth=2,  color='black')
    plt.plot(time_vec_ds, attn_norm, label="Avg Attention (Normalized)", linestyle='--', linewidth=2,  color='red')
    plt.title(f"Accuracy & Attention - Subject {subject_idx+1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Accuracy_Attention_subject_{subject_idx+1}.png"), dpi=300)
    # plt.close()

    # ---  Attention Curve ---
    num_heads = attn_weights.shape[1]  # should be 1 
    time_steps = attn_weights.shape[3]  # last dimension of attn_weights (T)
    plt.figure(figsize=(10, 4))
    for h in range(num_heads):
        # Average over batch and query dimensions → curve of shape (T,)
        head_avg_curve = np.mean(attn_weights[:, h, :, :], axis=(0,1))
        plt.plot(time_vec_ds, head_avg_curve, linewidth=1.5)
    plt.title(f"Attention Curve - Subject {subject_idx+1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Attention Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Headwise_Attention_subject_{subject_idx+1}.png"), dpi=300)
    # plt.close()


# Final Reporting
avg_acc = np.mean(all_accuracy)
avg_class_accuracy = np.mean(np.stack(all_class_accuracy, axis=1), axis=1)
avg_precision = np.mean(np.stack(all_precision, axis=1), axis=1)
avg_recall = np.mean(np.stack(all_recall, axis=1), axis=1)
avg_f1 = np.mean(np.stack(all_f1, axis=1), axis=1)
avg_overall_precision = np.mean(overall_precision)

print("\nAverage Accuracy:", avg_acc)
print("Average Accuracy per class:", avg_class_accuracy)
print("Average Precision per class:", avg_precision)
print("Average Recall per class:", avg_recall)
print("Average F1 per class:", avg_f1)
print("Average Overall Precision:", avg_overall_precision)


# Plot median ROC curve with IQR for all LOSO folds
# --- Set plot style ---
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# --- Interpolate all TPRs to common FPR grid ---
mean_fpr = np.linspace(0, 1, 100)
tprs = []
for i in range(len(all_fpr)):
    interp_tpr = np.interp(mean_fpr, all_fpr[i], all_tpr[i])
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
tprs = np.array(tprs)

# --- Compute statistics ---
mean_tpr = np.median(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
q25 = np.percentile(tprs, 25, axis=0)
q75 = np.percentile(tprs, 75, axis=0)

# --- Plot ---
plt.figure(figsize=(4, 4))
plt.plot(mean_fpr, mean_tpr, color='purple', lw=2, label=f'Median ROC AUC = {mean_auc:.3f})')
plt.fill_between(mean_fpr, q25, q75, color='purple', alpha=0.3, label='IQR')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=12, prop={'weight': 'bold'})
plt.tight_layout()


save_path = os.path.join(save_dir, "Median_ROC_Curve_iqr.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')


# Plot the acuracy and attention weights per sample on the same plot, with seperate y-axis scales (two rullers at the right and left)
bold_font = FontProperties(weight='bold', size=28)

plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# --- Process attention weights ---
subject_atten_curves = []  

for attn_weights in all_attention_weights:
    attn_avg_matrix = np.median(attn_weights, axis=(0, 1))  # (T, T)
    attn_curve = np.median(attn_avg_matrix, axis=0)         # (T,)
    
    # store the raw mean curve
    subject_atten_curves.append(attn_curve)
    
subject_atten_curves = np.stack(subject_atten_curves, 0) # (n_subjects, T)


# Median + IQR 
avg_attntion_curve = np.median(subject_atten_curves, axis=0)
attn_q25 = np.percentile(subject_atten_curves, 25, axis=0)
attn_q75 = np.percentile(subject_atten_curves, 75, axis=0)

# --- Process accuracy ---
subject_accs = np.stack(per_subject_sample_accuracy, axis=0)  # (n_subjects, T)
avg_sample_accuracy = np.median(subject_accs, axis=0)
acc_q25 = np.percentile(subject_accs, 25, axis=0)
acc_q75 = np.percentile(subject_accs, 75, axis=0)

# --- Plot ---
plt.figure(figsize=(9, 4))
ax = plt.gca()

# Accuracy (left y-axis)
ax.plot(time_vec_ds, avg_sample_accuracy, linewidth=2, color='black', label='Accuracy')
ax.fill_between(time_vec_ds, acc_q25, acc_q75, color='black', alpha=0.2)

ax.set_xlabel("Time (ms)", fontsize=13, fontweight='bold')
ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.02])
ax.tick_params(axis='both', labelsize=12)

# Collect legend handles
lines1, labels1 = ax.get_legend_handles_labels()

# --- Create a second y-axis for UNNORMALIZED attention (scaled by 1e3 in labels) ---
ax2 = ax.twinx()
# Plot raw data
ax2.plot(time_vec_ds, avg_attntion_curve, linewidth=2, color='darkred', label='Attention')
ax2.fill_between(time_vec_ds, attn_q25, attn_q75, color='red', alpha=0.2)

ax2.set_ylabel("Attention (×10³)", fontsize=13, fontweight='bold')
# Apply a custom formatter that multiplies the tick labels by 1000
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x * 1e3:.2f}"))
ax2.tick_params(axis='y', labelsize=12)

# Combine legends from both axes
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10, prop={'weight': 'bold'})

plt.tight_layout()

# Save
save_path = os.path.join(save_dir, "acc_att_curves.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')





