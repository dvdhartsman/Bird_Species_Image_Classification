# Bird Species Image Classification:
### Using Computer Vision to Identify 525 Different Bird Species
##### David Hartsman

![The majestic Puteketeke, New Zealand's "Bird of the Century"](./files/puteketeke.png)

### Overview
In this project, I utilized data from a Kaggle dataset containing roughly 90k images of 525 different bird species. I used tensorflow to create Convolusional Neural Networks to predict the correct species of the bird from an image. 

### Data
The data in this project were "RGB" images with dimensions of 224 pixels by 224 pixels. The images were curated to be of very high quality, and the birds themselves comprised roughly 50% of each image. 

![Example of Birds from the Data](./files/example_birds.jpg)

This provided the models with a substatial amount of "signal" to learn from, and should have aided in improving the accuracy of predictions. Keras has built in classes and functions that facilitated the creation of a data pipeline from the image directories to the model itself. That allowed me to use the built in functions to extract the numeric image data, scale it, and augment it, AND ALSO extract the correct labels from the data without having to explicitly code a labeling function. 

### Evaluation
Even though this data was not exceptionally large, it still required a long time to train each model iteration. I used both [Tensorboard](https://www.tensorflow.org/tensorboard) and [Weights and Biases](https://wandb.ai/site) to create dashboards that tracked model performance across training epochs. 

![Training Metrics from the First Model](./files/model_metrics.png)

![Training Metrics from the First Model](./files/accuracy.png)

Even when the first model made incorrect predictions, there was a clear proximity to the correct species. Take for example, this incorrect prediction:

![Mis-identified Bird Species](./files/incorrect_predictions.png)




### Conclusion
Due in part to the high number of class labels, I was only able to achieve an accuracy of roughly 43%. This model has been deployed on streamlit, and I encourage you to test the model's predictive ability with any bird image that you can find. In order to improve upon the model further, I believe that both more data and more computational resources would be required. Please feel free to explore the project in more detail by examining my [notebook](https://github.com/dvdhartsman/Bird_Species_Image_Classification/blob/main/Bird_Classification_1.ipynb), and contact me with any questions you may have.
