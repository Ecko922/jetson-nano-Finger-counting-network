# Jetson-nano-project: counting fingers

#Purpose: This network aims to recognize hand gestures that indicate different numbers.

#GUIDE:
This project includes a trained model that works. Thus, to operate this network, you just need to activate all the codes and load the pre-trained model in the notebook. However, if you decide to add more categories or train the model even more, the notebook allows you to make changes to the code or add more images to the datebase.

#Requirements:
- Jetson Nano, or another operating computer. 
- An USB camera
- Python3

### CODE RUNTHROUGH ###

#Camera#
This sets the size of the images and starts the camera. Make sure that the correct camera type is selected for execution (USB). 

#Task#
It defines TASK and CATEGORIES parameters here, also the dataset. 

#Data Collection#
This cell sets up the collection mechanism to count the images and produce the user interface. The widget built is the data_collection_widget. 

#Model#
This block is where the neural network is defined. First, the GPU device is chosen with the statement:

device = torch.device('cuda')
The model is set to the ResNet-18 model for this project. 

model = torchvision.models.resnet18(pretrained=True)

#Live Execution#
This code block sets up threading to run the model in the background so that you can see the live camera feed and visualize the model performance in real time. It also includes the code that defines how the outputs from the neural network are categorized. The network produces some value for each of the possible categories. 

output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()

#Training and Evaluation#
The training code cell sets the hyper-parameters for the model training and loads the images for training or evaluation. The model determines a predicted output from the loaded input image. The difference between the predicted output and the actual label is used to calculate the "loss". If the model is in training mode, the loss is backpropagated into the network to improve the model. The widgets created by this code cell include the option for setting the number of epochs to run. One epoch is a complete cycle of all images through the trainer. This code cell may take several seconds to execute.

#Display the Interactive Tool#
This cell should display the full tool for you to work with. 
