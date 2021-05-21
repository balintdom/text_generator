from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf



class Dataset:
    
    def __init__(self, text):
        """
            create a dataset for character based language models from a text dataset
        """
        
        # Batch size
        self.BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        self.BUFFER_SIZE = 10000

        # sequence length of the inputs and outputs
        self.SEQ_LENGTH = 100
        
        vocab = sorted(set(text))

        #define the char to id and the id to char functions
        self.ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(vocab))

        self.vocab = self.ids_from_chars.get_vocabulary()
        self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.vocab, invert=True)

        
        all_ids = self.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        # create the 100 long char sequences
        sequences = ids_dataset.batch(self.SEQ_LENGTH+1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)

        #define the tensorlfow dataset
        self.dataset = (
            dataset
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        
    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
