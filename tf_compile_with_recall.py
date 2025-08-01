import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer="zeros")
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int64)
        y_pred_labels = tf.cast(y_pred_labels, tf.int64)

        for i in range(self.num_classes):
            true_i = tf.equal(y_true, i)
            pred_i = tf.equal(y_pred_labels, i)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(true_i, pred_i), self.dtype))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(true_i, tf.logical_not(pred_i)), self.dtype))

            # Обновим всю переменную через tf.tensor_scatter_nd_add
            indices = [[i]]
            self.true_positives.assign(tf.tensor_scatter_nd_add(self.true_positives, indices, [tp]))
            self.false_negatives.assign(tf.tensor_scatter_nd_add(self.false_negatives, indices, [fn]))

    def result(self):
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        return tf.reduce_mean(recall)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


# Данные
np.random.seed(42)
X = np.random.rand(10, 5)
y = np.random.randint(0, 4, size=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Компиляция с кастомной метрикой
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=[MacroRecall(num_classes=4)]
)

# Обучение
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
