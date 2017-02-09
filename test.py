from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, merge
from keras.models import Model, Sequential
from keras.utils.visualize_util import plot

# first, let's define a vision model using a Sequential model.
# this model will encode an image into a vector.
vision_model = Sequential(name='vision_model')
vision_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(3, 224, 224), dim_ordering="th"))
vision_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering="th"))
vision_model.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering="th"))
vision_model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering="th"))
vision_model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering="th"))
vision_model.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering="th"))
vision_model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering="th"))
vision_model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering="th"))
vision_model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering="th"))
vision_model.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering="th"))
vision_model.add(Flatten())


# now let's get a tensor with the output of our vision model:
image_input = Input(shape=(3, 224, 224))
encoded_image = vision_model(image_input)

# next, let's define a language model to encode the question into a vector.
# each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# let's concatenate the question vector and the image vector:
merged = merge([encoded_question, encoded_image], mode='concat')

# and let's train a logistic regression over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# this is our final model:
vqa_model = Model(input=[image_input, question_input], output=output,name='vqa_model')

# the next stage would be training this model on actual data.


from keras.layers import TimeDistributed

video_input = Input(shape=(100, 3, 224, 224))
# this is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

# this is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(input=question_input, output=encoded_question,name='question_encoder')

# let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# and this is our video question answering model:
merged = merge([encoded_video, encoded_video_question], mode='concat')
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(input=[video_input, video_question_input], output=output)



plot(vqa_model, to_file='model.png',show_layer_names=True,show_shapes=True)


#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot
#
#SVG(model_to_dot(video_qa_model).create(prog='dot', format='svg'))