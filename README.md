# Learning Time series analysis

## 1. Training Timeseries classifier on FordA dataset from the UCR/UAE.
source: keras https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

## 2. gait analysis
We build a multimodal neural network with three inputs (side-view series, front-view series, and static features) that outputs a binary diagnosis. First, we prepare the data, then define separate subnetworks for each input, concatenate their outputs, and train the combined model with appropriate callbacks. Finally, we evaluate on a held-out test set.

## Data Preparation
* Load the CSV files: We use pandas to read the side-view and front-view CSVs. Each row has (subject, frame, label, plus features).
* Sort and encode: For each subject, we sort by frame. The Direction column is categorical, so we encode it as numeric (e.g. left=0, right=1, stationary=2 for side view; towards=0, backwards=1 for front).
* Extract numeric features:
* Side features (300×9): We take the 8 angle-related columns and the encoded direction. This gives 9 features per timestep. We pad or truncate each subject’s series to length 300.
* Front features (249×4): We take the 3 front-view features (R_Adduction_Abduction, L_Adduction_Abduction, Step_Width) plus the encoded direction, then pad/truncate to 249 frames.
* Static (univariate) features (3 per subject): We compute three summary features per subject (for example, cadence, stride length, step length,velocity). This yields a (53, 3) array aligned with subjects.
* Align labels: We extract the binary label for each subject (0 = non-autistic, 1 = autistic) from one of the datasets (they should agree).
* Train-test split: We split the 53 subjects 80/20 (e.g. 42 train, 11 test) to ensure no subject overlap, using sklearn.model_selection.train_test_split.

## Model Architecture

We define a three-branch neural network. Each branch processes one input modality, then we concatenate their outputs and add dense layers for classification.
* Side-view branch: Input shape (300, 9). We apply 1D convolutional layers to capture temporal patterns of the 9-channel sequence. For example:
* Conv1D(32 filters, kernel=3, activation='relu') + MaxPool
* Conv1D(64 filters, kernel=3, activation='relu') + GlobalAveragePooling1D
This yields a fixed-length feature vector for the side-view input.
* Front-view branch: Input shape (249, 4). We similarly use Conv1D layers (with smaller filters, e.g. 16 and 32 filters) and pooling, ending in GlobalAveragePooling.
* Univariate branch: Input shape (3,). We use a small Dense network, e.g. Dense(32, relu) → Dropout → Dense(16, relu).
These three outputs are concatenated and fed into further Dense layers for classification. Finally, a single-unit sigmoid outputs the probability of autism. We compile with the Adam optimizer and binary cross-entropy loss.
