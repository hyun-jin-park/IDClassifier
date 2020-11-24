import torch
from torch.nn import CrossEntropyLoss, Linear
import torchvision.models as models


class BasicClassifier (torch.nn.Module):
    def __init__(self,  n_class, hidden=512, pretrained=True, freeze_head=True):
        super().__init__()
        # self.backbone = models.resnet101(pretrained=pretrained)
        self.backbone = models.resnet50(pretrained=pretrained)
        if freeze_head :
            for param in self.backbone.parameters():
                 param.requires_grad = False
        self.backbone.fc = Linear(self.backbone.fc.in_features, hidden)
        self.fc = Linear(hidden, n_class)
        self.loss = CrossEntropyLoss()

    def forward(self, input_images, labels=None):
        loss = None
        outputs = self.backbone(input_images)
        outputs = torch.nn.functional.relu(outputs)
        outputs = self.fc(outputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        if labels is not None:
            loss = self.loss(outputs, labels)
        return outputs, loss


