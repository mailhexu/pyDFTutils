from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.trainer import Trainer

dataset = StructureData(
    structures=list_of_structures,
    energies=list_of_energies,
    forces=list_of_forces,
    stresses=list_of_stresses, # THe unit is GPa
    magmoms=list_of_magmoms,
)
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset, batch_size=32, train_ratio=0.9, val_ratio=0.05
)
trainer = Trainer(
    model=chgnet,
    targets="efsm",
    optimizer="Adam",
    criterion="MSE",
    learning_rate=1e-2,
    epochs=50,
    use_device="cuda",
)

trainer.train(train_loader, val_loader, test_loader)

