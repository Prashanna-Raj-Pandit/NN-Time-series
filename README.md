# Learning Time series analysis

## 1. Training Timeseries classifier on FordA dataset from the UCR/UAE.
source: keras https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

## 2. gait analysis
We build a multimodal neural network with three inputs (side-view series, front-view series, and static features) that outputs a binary diagnosis. First, we prepare the data, then define separate subnetworks for each input, concatenate their outputs, and train the combined model with appropriate callbacks. Finally, we evaluate on a held-out test set.

## Data Preparation
* Load the CSV files: We use pandas to read the side-view and front-view CSVs. Each row has (subject, frame, label, plus features).
Sort and encode: For each subject, we sort by frame. The Direction column is categorical, so we encode it as numeric (e.g. left=0, right=1, stationary=2 for side view; towards=0, backwards=1 for front).
Extract numeric features:
Side features (300×9): We take the 8 angle-related columns and the encoded direction. This gives 9 features per timestep. We pad or truncate each subject’s series to length 300.
Front features (249×4): We take the 3 front-view features (R_Adduction_Abduction, L_Adduction_Abduction, Step_Width) plus the encoded direction, then pad/truncate to 249 frames.
Static (univariate) features (3 per subject): We compute three summary features per subject (for example, mean step width, mean right hip angle, mean left hip angle). This yields a (53, 3) array aligned with subjects.
Align labels: We extract the binary label for each subject (0 = non-autistic, 1 = autistic) from one of the datasets (they should agree).
Train-test split: We split the 53 subjects 80/20 (e.g. 42 train, 11 test) to ensure no subject overlap, using sklearn.model_selection.train_test_split.


