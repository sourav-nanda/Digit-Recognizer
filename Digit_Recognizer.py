from tensorflow.keras.models import load_model
import gradio as gr

model_l=load_model('digit_model_deepnote.h5')

def digit_classifier(image):
    image=image.reshape(1,28,28,1)
    pred=model_l.predict(image).flatten()
    labels=[j for j in range(10)]
    return {labels[i]: float(pred[i]) for i in range(10)}

if __name__ == '__main__':
	digit_classifier(image)

interface=gr.Interface(fn=digit_classifier,
                       inputs=gr.inputs.Sketchpad(shape=(28,28)),
                       outputs='label',
                       live=True,
                       capture_session=True,
                       title='Handwritten Digit Recognizer',
                       description='Write any digit and see the predictions in terms of labels',
                       server_name="0.0.0.0")
interface.launch(share=True)
