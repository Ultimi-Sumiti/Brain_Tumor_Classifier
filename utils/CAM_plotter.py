from model.model_v1 import *
from utils.plot_utils import *

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]


mask_paths = [
    "./data/masks/brisc2025_test_00097_gl_co_t1.png",
    "./data/masks/brisc2025_test_00257_me_ax_t1.png",
    "no-tm",
    "./data/masks/brisc2025_test_00997_pi_sa_t1.png"
]

sample_paths = [
    "./data/test/glioma/brisc2025_test_00097_gl_co_t1.jpg",
    "./data/test/meningioma/brisc2025_test_00257_me_ax_t1.jpg",
    "./data/test/no_tumor/brisc2025_test_00579_no_ax_t1.jpg",
    "./data/test/pituitary/brisc2025_test_00997_pi_sa_t1.jpg"
]


# Get cams for the samples.
cams = []
predicted = []
for img_path in sample_paths:
    cam, class_idx = get_cam(img_path, model, plot=False)
    cams.append(cam)
    predicted.append(class_names[class_idx])


# Plot.
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, name in zip(range(len(mask_paths)), class_names):
    if mask_paths[idx] != "no-tm":
        mask = Image.open(mask_paths[idx])
    else:
        mask = None
    orig = Image.open(sample_paths[idx])
    cam = cams[idx]
    predict = predicted[idx]

    ax_mask = axes[0, idx]
    ax_out = axes[1, idx]

    if mask is not None:
        ax_mask.imshow(mask, cmap='gray')
    else:
        ax_mask.imshow(np.zeros_like(orig), cmap='gray')
    ax_mask.set_title(f'Mask - {name}', fontsize=20)
    ax_mask.axis('off')

    ax_out.imshow(orig, cmap="gray")
    ax_out.imshow(cam, cmap="jet", alpha=0.5)

    ax_out.set_title(f'Predicted - {predict}', fontsize=20)
    ax_out.axis('off')

plt.tight_layout()
plt.show()
#plt.savefig("heatmap.png")

# Test on one image.
#ckph_path = "./lightning_logs/version_73/checkpoints/best-checkpoint-epoch=06-val_loss=0.05.ckpt"
#model = ResNetFineTuner.load_from_checkpoint(ckph_path).cpu()
#image_path = "./data/test/meningioma/brisc2025_test_00259_me_ax_t1.jpg"
#cam_output = get_cam(image_path, model)
