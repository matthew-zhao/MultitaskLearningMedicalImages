import numpy as np

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def coalesce_predictions(y_pred, y_true, y_levels): 
    """
    Coalesce image-level predictions into patient-level.
    """
    by_level = {'y_true': {}, 'y_pred': {}}
    for lvl in np.unique(y_levels): 
        by_level['y_pred'][lvl] = [] 
        by_level['y_true'][lvl] = [] 
    for i, pred in enumerate(y_pred): 
        by_level['y_pred'][y_levels[i]].append(pred)
        by_level['y_true'][y_levels[i]].append(y_true[i])
    for lvl in by_level['y_pred'].keys():
        by_level['y_pred'][lvl] = np.mean(by_level['y_pred'][lvl], axis=0)
        by_level['y_true'][lvl] = np.max(by_level['y_true'][lvl])
    y_pred_list = []
    y_true_list = []
    for lvl in by_level['y_pred'].keys():
        y_pred_list.append(by_level['y_pred'][lvl])
        y_true_list.append(by_level['y_true'][lvl])
    y_pred = np.vstack(y_pred_list)
    y_true = np.asarray(y_true_list)
    return y_pred, y_true

def u_ones(label):
    """
    convert all blanks to 0.0s and the -1 (uncertain label) to 1.0 (positive)
    """
    if not label:
        return 0

    if (isinstance(label, float) or isinstance(label, int)) and label < 0:
        return 1

    return label

def u_zeros(label):
    """
    convert all blanks to 0.0s and the -1 (uncertain label) to 0.0 (negative)
    """
    if not label:
        return 0

    if (isinstance(label, float) or isinstance(label, int)) and label < 0:
        return 0

    return label

def u_ignore(label):
    """
    convert all blanks to 0.0s and the -1 (uncertain label) to None (we will handle to ignore)
    """
    if not label:
        return 0

    if (isinstance(label, float) or isinstance(label, int)) and label < 0:
        return None

    return label

def u_multiclass(label):
    """
    convert all blanks to 0.0s and the -1 (uncertain label) to None (we will handle to ignore)
    """
    if not label:
        return 0

    if (isinstance(label, float) or isinstance(label, int)) and label < 0:
        return 2

    return label

