# TODO: Map value to midi pitch

label_to_value_mapping = {
    # 'A': 0,
    # 'B': 1,
    # 'C': 2,
}
value_to_label_mapping = {
    # 0: 'A',
    # 1: 'B',
    # 2: 'C',
}


def label_to_value(label):
    return label_to_value_mapping[label]


def value_to_label(value):
    return value_to_label_mapping[value]
