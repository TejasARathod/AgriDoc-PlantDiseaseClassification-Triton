import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Directory and dataset name
data_dir = "train"
dataset_name = "trial100"

# Avoid redefining the module name as a variable
# Avoid "dir" as a variable name — it's a Python built-in

# Delete dataset if it already exists (optional but clean)
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

# Load the dataset
dataset = fo.Dataset.from_dir(
    data_dir,
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    name=dataset_name,
)
dataset.persistent = True

# View summary info
print(dataset)
print(dataset.head())

# Load a ResNet50 model
model = foz.load_zoo_model("resnet50-imagenet-torch")

# Confirm model supports embeddings
print("Has embeddings:", model.has_embeddings)

# Compute and assign embeddings
embeddings = dataset.compute_embeddings(model=model)
print("Embeddings shape:", embeddings.shape)

# Visualize using FiftyOne Brain (using precomputed embeddings)
fob.compute_visualization(
    dataset,
    embeddings=embeddings,
    num_dims=2,
    brain_key="image_embeddings",
    verbose=True,
    seed=51,
)

# Launch FiftyOne App — do this LAST
session = fo.launch_app(dataset, port=5151)

# Prevent script from exiting (keeps session alive)
session.wait()

