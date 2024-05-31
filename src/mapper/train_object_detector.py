import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one for our dataset
num_classes = 2  # 1 class (door) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training code (simplified)
# dataset = YourCustomDataset()
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Define optimizer and learning rate
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for images, targets in data_loader:
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

# Save the fine-tuned model
# torch.save(model.state_dict(), "fine_tuned_model.pth")
