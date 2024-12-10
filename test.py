import torch
from model_ResNet import AHCR
from dataset import data_loader, DvX_dataset_collate
from loss_func import BCELoss
from torch.utils.data import DataLoader
from utils import confidence_weighted_view_fusion
from get_ap import AveragePrecisionMeter

# Testing parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = [224, 224]
batch_size = 64  # Use a smaller batch size for testing
test_annotation_path = '/home/user4/DvXray/DvXray_test.txt'  # Path to the test annotation file

# Class-to-label mapping
class_labels = ['Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors', 'Lighter', 'Battery', 'Bat', 'Razor_blade',
                'Saw_blade', 'Fireworks', 'Hammer', 'Screwdriver', 'Dart', 'Pressure_vessel']

def test():
    # Load the model
    model = AHCR(num_classes=15).to(device)
    checkpoint_path = './checkpoint/ep008_ResNet_checkpoint.pth.tar'  # Path to the model checkpoint

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"Model checkpoint not found: {checkpoint_path}")
        return

    # Define the loss function
    criterion = BCELoss().to(device)

    # Load test data
    with open(test_annotation_path) as f:
        test_lines = f.readlines()

    test_loader = DataLoader(
        data_loader(test_lines, input_shape),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DvX_dataset_collate
    )

    # Model inference
    model.eval()
    ap_meter = AveragePrecisionMeter()
    ap_meter.reset()

    with torch.no_grad():
        for i, (img_ols, img_sds, gt_s) in (enumerate(test_loader)):
            img_ols = img_ols.to(device)
            img_sds = img_sds.to(device)
            gt_s = gt_s.to(device)

            # Forward pass
            ol_output, sd_output = model(img_ols, img_sds)

            # Apply fusion method
            prediction = confidence_weighted_view_fusion(torch.sigmoid(ol_output), torch.sigmoid(sd_output))
            print(f"Batch {i + 1}/{len(test_loader)}")

            # Add predictions to AP calculator
            ap_meter.add(prediction.data, gt_s)

    # Compute AP for each class and the mean AP (mAP)
    each_ap = ap_meter.value()
    map_score = 100 * each_ap.mean()

    print("AP for each class:")
    for i, label in enumerate(class_labels):
        print(f"{label}: {each_ap[i] * 100:.2f}%")

    print(f"\nmAP: {map_score:.2f}%")

if __name__ == "__main__":
    test()
