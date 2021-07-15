import torch
from pytorchvideo.models.mot.tracker import JDETracker

file_name = "jde_dets.pt"
loaded_save_dict = torch.load(file_name)
# print(loaded_save_dict)

tracker = JDETracker()
print(dir(tracker))

for i in range(10):
    tracker.update(loaded_save_dict[str(i)][:, :5], loaded_save_dict[str(i)][:, 6:])

print("End of run")



