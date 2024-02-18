# Bird Species Image Classification:
### Using Computer Vision to Identify 525 Different Bird Species
##### David Hartsman

![The majestic Puteketeke, New Zealand's "Bird of the Century"](./files/puteketeke.png)

### Overview
In this project, I created a Convolutional Neural Network modeled on data from a Kaggle dataset containing roughly 90k images of 525 different bird species. I used tensorflow to create the Convolutional Neural Networks. The purpose was to maximize the model's ability to predict the correct species of the bird from an image. 

### Data
The data in this project were "RGB" (3-channel, red-green-blue color) images with dimensions of 224 pixels by 224 pixels. The images were curated to be of very high quality.

![Example of Birds from the Data](./files/example_birds.jpg)

This provided the models with a substatial amount of "signal" to learn from, and aided in improving the accuracy of predictions. Keras comes with built-in classes and functions that facilitated the creation of a data pipeline directly from the local image directories to the model itself. That allowed me to use the built-in functions to extract the numeric image data, scale it, and augment it, and also extract the correct labels from the data without having to explicitly code a labeling function. 

### Evaluation
This data was large enough, in combination with the depths and widths of model iterations, to still require substantial time to train each model iteration. I used both [Tensorboard](https://www.tensorflow.org/tensorboard) and [Weights and Biases](https://wandb.ai/site) to create dashboards that tracked model performance across training epochs. The models also return dictionaries containing training logs for the metrics that were being tracked over each epoch of training. 

![Training Metrics from the First Model](./files/model_metrics.png)

![Training Metrics from the First Model](./files/accuracy.png)

In the early stages of modeling, accuracy was topping out at around 40%. 
Even when the first model made incorrect predictions, there was a clear proximity to the correct species. Take for example, this incorrect prediction:

![Mis-identified Bird Species](./files/incorrect_predictions.png)




### Conclusion
I was only able to achieve an accuracy of roughly 43% on my first model, however I was able to eventually achieve much better results by leveraging the EfficientNetB0 architecture. I achieved accuracy scores on test data of around 98% using this architecture. I additionally tested the model on "WILD" images from the internet, and the model still performed very well. The process for preparing this "wild" data for model predictions required resizing it from its natural pixel ratio to (224, 224, 3) and then accounting for the model being trained on batches needed to be addressed as well by using np.expand_dims([image_file], 0). 

![Representative of the "Wild" Image Preparation](./files/abbotts_compare.jpg)

This model has been deployed on streamlit, and I encourage you to test the model's predictive ability with any bird image that you can find. In order to improve upon the model further, I believe that both more data and more computational resources would be required. Please feel free to explore the project in more detail by examining my [notebook](https://github.com/dvdhartsman/Bird_Species_Image_Classification/blob/main/Bird_Classification_1.ipynb), and contact me with any questions you may have.
