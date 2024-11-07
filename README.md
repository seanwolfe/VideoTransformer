# VideoTransformer
Video Transformer for asteroid detection using VideoMAE, masked autoencoding for pretraining. The video transformer
is trained on synthetic examples from the Asteroid Image Generator repository. Some files contain a classifier, which
determines if an asteroid is present in the stack or not, while others contain a regressor, which predicts the
center of the asteroid in the first frame. Each of these has there respective config file.
