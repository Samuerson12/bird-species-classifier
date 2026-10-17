# Bird species classifier (CUB-200)

COS30082 Applied Machine Learning, Assignment 1
Samuel Chang Hao Jun, 102775957

A Streamlit web app that identifies 200 bird species from a photo. Upload an image and it shows the five most likely species with their probabilities.

The model is EfficientNetB0 pretrained on ImageNet, used as a frozen feature extractor, with a softmax regression classifier trained on the CUB-200 training images. On the test set it reaches 70.76% top-1 accuracy and 70.20% average accuracy per class.

## Files

```
app.py                      the Streamlit app
requirements.txt            Python packages (versions match the Colab training environment)
.streamlit/config.toml      colour theme and upload size limit
model/bird_classifier.keras the trained model, exported by notebook 4
model/class_names.json      species names, index i is class i
model/model_info.json       test results shown in the sidebar
```

The three files in `model/` come from the `outputs/deploy` folder on Google Drive after running notebook 4.

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository
2. Sign in at share.streamlit.io with GitHub and choose Create app
3. Pick the repository, branch `main` and main file `app.py`
4. Under Advanced settings, choose Python 3.12
5. Deploy. The first start takes a few minutes while TensorFlow installs
