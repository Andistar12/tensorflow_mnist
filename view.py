import json
import os
import matplotlib.pyplot as plt

NETWORKS_LOC = "./networks"
networks = dict()

# Load in all available networks as JSON
for file in os.listdir(NETWORKS_LOC):
    if file.endswith(".json"):
        with open(os.path.join(NETWORKS_LOC, file)) as json_file:
            data = json.load(json_file)
            name = data.get("name", "")
            if name is "":
                continue
            if data.get("train_data", "") is not "":
                networks[name] = data["train_data"]

print("Found {0} items in directory {1}".format(len(networks), NETWORKS_LOC))
print(" ".join(networks.keys()))
net = input("Enter name to view: ")

if net in networks:
    network = networks[net]
    window, axes = plt.subplots(1)
    x_axis = list(range(1, len(network["loss"]) + 1))

    axes.plot(x_axis, network["loss"], "bo-", label="Train Loss")
    axes.plot(x_axis, network["acc"], "b+-", label="Train Acc")
    axes.plot(x_axis, network["val_loss"], "go-", label="Test Loss")
    axes.plot(x_axis, network["val_acc"], "g+-", label="Test Acc")

    axes.set_title(net + " performance")
    axes.set_xlabel("Epoch")
    axes.legend(loc='upper left')
    window.show()

    input("Press enter to quit")
else:
    print("No network found for given input. Quitting")
