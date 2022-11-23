def shape_printer(name, tensor):
    print(f"Shape of {name}: {tensor.shape}")


def id_ood_printer(id_classes, ood_classes):
    print(f"\nid Classes: {id_classes[:2]}... \n\n OOD classes: {ood_classes[:2]}...")


def name_printer(name):
    print("\n" * 2, "-" * 30, name, "-" * 30, "\n")


def mean_std_printer(mean, std, runs):
    print(f"Runs: {runs}\t\tMEAN: {mean}\t\t STD: {std}")
