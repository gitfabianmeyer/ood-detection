def shape_printer(name, tensor):
    print(f"Shape of {name}: {tensor.shape}")


def id_ood_printer(id_classes, ood_classes):
    print(f"id Classes: {id_classes}\n\n OOD classes: {ood_classes}")


def name_printer(name):
    print("\n" * 2, "-" * 30, name, "-" * 30, "\n")
